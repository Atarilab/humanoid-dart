import gc
import traceback
import numpy as np
import os
import shutil
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import torch
import wandb
import yaml

from humanoid_dart.dart.utils.trajectory_utils import log_memory
from humanoid_dart.dart.pipeline_config import PipelineConfig, parse_pipeline_args
from humanoid_dart.dart.augment_trajectory import augment_trajectory_data
from humanoid_dart.dart.evo_task_config import get_task_config
from humanoid_dart.dart.archive import Elite, EliteArchive, plot_diffusion_training_subset, plot_targets_vs_achieved, plot_targets_vs_achieved_6d, plot_archive_6d
from humanoid_dart.dart.utils.logger import IterationLogger
from humanoid_dart.dart.utils.trajectory_utils import weighted_sum_trajectory, batch_elites, compute_batch_metrics, regenerate_sim_data, clean_batch_rollouts, overwrite_with_rollouts, load_base_motions, ensure_rl_schema
from humanoid_dart.dart.runner import run_batch_simulation_multi_gpu, run_policy_training
from humanoid_dart.dart.multi_device import get_available_devices
from humanoid_dart.planner.motion_generator import MotionGenerator

DEFAULT_PLANNER_CONFIG = Path(__file__).resolve().parent.parent / "planner" / "config" / "config.yaml"


def _write_run_planner_config(src: Path, out_root: Path) -> Path:
    """Copy the planner config into the run dir, pointing its save_dir there.

    The planner keeps its normalization stats under ``save_dir``; keeping them
    per run means every run computes them from its own base motions instead of
    reusing a stale ``norm_stats.npz`` left by an earlier run.
    """
    with open(src) as f:
        planner_cfg = yaml.safe_load(f)
    planner_cfg.setdefault("training", {})["save_dir"] = str(out_root / "planner") + "/"
    dst = out_root / "planner_config.yaml"
    with open(dst, "w") as f:
        yaml.safe_dump(planner_cfg, f, sort_keys=False)
    return dst


def main():
    cfg = parse_pipeline_args()
    if cfg.replanning.enabled:
        raise NotImplementedError("Closed-loop replanning is not part of this release; set replanning.enabled: false.")

    out_root = Path(cfg.out_dir).resolve()
    if out_root.exists():
        # Only wipe directories that a previous pipeline run created.
        if any(out_root.iterdir()) and not (out_root / "run_config.yaml").exists():
            raise FileExistsError(f"{out_root} exists and is not a previous run directory; pick another out_dir.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # Save resolved config for reproducibility
    cfg.save_yaml(out_root / "run_config.yaml")

    # WandB run names are "<name>_Gen<k>", defaulting to the out_dir name.
    _dir_prefix = cfg.wandb.name or out_root.name

    base_motion_path = Path(cfg.base_motion).resolve()


    # Load base motion(s)
    all_base_data, base_motion_path = load_base_motions(base_motion_path)

    # Use first motion as "primary" for dimension checks / initializations
    base_data = all_base_data[0]
    base_joint_pos = base_data['joint_pos']

    # Load Task Config
    evo_task_name = cfg.evo_task
    if not evo_task_name:
        raise ValueError("evo_task must be set when running the evolutionary pipeline.")
    task_config = get_task_config(evo_task_name)

    devices = get_available_devices(cfg.num_gpus)
    device = devices[0]
    effective_num_gpus = len(devices) if device != "cpu" else 1
    print(f"[INFO] Resolved pipeline devices: {devices}")
    
    # Initialize Diffusion Model
    cfg_path = str(_write_run_planner_config(
        Path(cfg.diffusion.config_path) if cfg.diffusion.config_path else DEFAULT_PLANNER_CONFIG,
        out_root,
    ))
    print(f"[INFO] Initializing Diffusion MotionGenerator with config: {cfg_path}")
    diff_generator = MotionGenerator(config_path=cfg_path, device=device, num_gpus=effective_num_gpus)

    # Compute task params for each base trajectory using the same metric functions
    # used to build elite_targets, so the initial fit is goal-conditioned.
    base_task_params = []
    for traj_data in all_base_data:
        pseudo = {k: (v[:, np.newaxis] if isinstance(v, np.ndarray) and v.ndim >= 2 else v)
                  for k, v in traj_data.items()}
        T_base = traj_data['joint_pos'].shape[0]
        start_idx = int(T_base * 0.5)
        vec = [float(dim.metric_fn(pseudo, start_idx)[0]) for dim in task_config.dimensions]
        base_task_params.append(vec)

    print(f"[INFO] Fitting diffusion model on {len(all_base_data)} base trajectories...")
    last_ckpt_path = diff_generator.fit(
        data_source=all_base_data,
        epochs=cfg.diffusion.initial_epochs,
        save_path=str(Path(cfg.out_dir) / "diffusion_initial.pt"),
        checkpoint=cfg.diffusion.checkpoint,
        calc_stats=False,
    )

    all_base_data = [ensure_rl_schema(d) for d in all_base_data]

    current_checkpoint = cfg.checkpoint_file

    if current_checkpoint is None:
        print("\n=== No checkpoint provided. Training initial policy from scratch on base motion... ===")
        
        new_checkpoint = run_policy_training(
            task_name=cfg.task,
            motion_file=str(base_motion_path),
            device=device,
            resume_checkpoint=current_checkpoint,
            num_envs=cfg.training.num_envs,
            max_iterations=cfg.training.initial_max_iterations,
            wandb_project=cfg.wandb.project,
            run_name=f"{cfg.wandb.name}_init_train" if cfg.wandb.name else "init_train",
            num_gpus=effective_num_gpus,
        )
        
        if new_checkpoint:
            current_checkpoint = new_checkpoint
            print(f"[INFO] Initial training complete. Checkpoint: {current_checkpoint}")
        else:
            raise RuntimeError("Failed to find checkpoint after initial training.")

    # Initialize elites archive (persists across generations)
    elites = EliteArchive(task_config=task_config)
    
    # Load previous elites if requested
    if cfg.load_elites:
        elites.load(cfg.load_elites)

    # Seed archive with base trajectories so they are included in diffusion training
    elites.seed_from_trajectories(all_base_data)

    # Cached diffusion training subset – only resampled when diffusion resets
    diff_subset: list | None = None

    for generation in range(cfg.generations):
      try:
        # Init WandB for this generation
        if cfg.wandb.project:
             if wandb.run is not None:
                 wandb.finish()
             
             wandb.init(
                project=cfg.wandb.project, 
                entity=cfg.wandb.entity, 
                name=f"{_dir_prefix}_Gen{generation}", 
                config=cfg.to_dict()
            )

        # When reset_every > 0, re-downsample once per generation so
        # newly discovered elites are included in the next training round.
        # When reset_every == 0, diff_subset is reset every iteration below.
        if cfg.diffusion.reset_every:
            diff_subset = None

        print(f"\n\n################################################")
        print(f"### GENERATION {generation+1}/{cfg.generations} ###")
        print(f"################################################\n")

        if cfg.curriculum == "thompson" and generation > 0:
            elites.decay_thompson_posteriors(gamma=0.9)
            print("[Curriculum] Thompson posterior decay applied (gamma=0.9)")
        
        gen_dir = out_root / f"gen_{generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting Evolutionary Pipeline (Gen {generation})")

        new_elites_count_cumulative = 0
        
        # Main Loop
        iterations = 2 if generation == -1 else cfg.iterations
        for iteration in range(iterations):
            print(f"\n=== Iteration {iteration+1}/{iterations} (Gen {generation}) ===")
            iter_dir = gen_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(exist_ok=True)

            logger = IterationLogger(
                generation=generation,
                iteration=iteration,
                num_samples=cfg.num_samples,
            )
            logger.start_timer()
            logger.archive_size_before = len(elites)

            log_memory(f"iter-{iteration}-start")
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                logger.cuda_free_gb = free / 1e9
                logger.cuda_total_gb = total / 1e9
                print(f"[CUDA] free {free/1e9:.2f} GB / total {total/1e9:.2f} GB")
            
            aug_files = []
            sources = []
            batch_targets = []
            traj_lengths = []  # Real trajectory lengths (before padding)
            
            # Determine base trajectory length and max generation length
            base_traj_len = base_joint_pos.shape[0]
            max_traj_len = cfg.evolution.max_traj_length if cfg.evolution.max_traj_length else base_traj_len
            variable_length = (max_traj_len != base_traj_len)
            
            # 1. Generate Trajectories
            print(f"Generating {cfg.num_samples} trajectories...")
            
            gen_start_time = time.time()
            
            # Save curriculum plot once per iteration (or generation)
            curriculum_plot_path = None
            if cfg.curriculum != "random":
                curriculum_plot_path = str(iter_dir / "curriculum_map.png")
            
            # Batch sample targets for efficiency
            if cfg.curriculum == "thompson":
                tv, bins = cast(
                    Tuple[List[Dict[str, float]], List[Tuple[int, ...]]],
                    elites.get_target_values(
                        curriculum=cfg.curriculum,
                        output_plot_path=curriculum_plot_path,
                        num_samples=cfg.num_samples,
                        return_bins=True,
                    ),
                )
                all_target_values = tv
                sampled_bin_tuples = bins
            else:
                all_target_values = cast(
                    List[Dict[str, float]],
                    elites.get_target_values(
                        curriculum=cfg.curriculum,
                        output_plot_path=curriculum_plot_path,
                        num_samples=cfg.num_samples,
                        return_bins=False,
                    ),
                )
                sampled_bin_tuples = []
            
            # Variable-length trajectories: the diffusion model determines
            # the real trajectory length via goal-stop. All arrays are padded
            # to max_traj_len; traj_lengths records the actual useful frames.
            
            # --- Batch Diffusion Generation ---
            # Pre-generate all diffusion samples in one go
            
            # Helper to keep data in memory instead of disk
            generated_data_batch: List[Optional[Dict[str, Any]]] = [None] * cfg.num_samples
            
            batch_indices = []
            batch_init_cond_robot = []
            batch_init_cond_obj = []
            batch_goals = []
                 
            # Prepare all conditions first
            for idx, t_val in enumerate(all_target_values):
                if random.random() < cfg.evolution.diffusion_probability:
                    batch_indices.append(idx)
                    
                    # Select Base
                    closest = elites.get_closest(t_val)
                    if closest:
                        sel_base = closest.data
                    else:
                        sel_base = all_base_data[random.randint(0, len(all_base_data)-1)]
                    
                    H = diff_generator.data_cfg.get("state_history", 1)
                    base_bp = sel_base['body_pos_w']
                    base_bq = sel_base['body_quat_w']
                    base_jp = sel_base['joint_pos']
                    
                    r_hist = np.concatenate([base_bp[:H,0,:], base_bq[:H,0,:], base_jp[:H,:]], -1)
                    batch_init_cond_robot.append(r_hist)
                    
                    if 'object_pos_w' in sel_base:
                        o_hist = np.concatenate([sel_base['object_pos_w'][:H,:], sel_base['object_quat_w'][:H,:]], -1)
                        batch_init_cond_obj.append(o_hist)
                        
                    batch_goals.append([float(t_val[d.name]) for d in task_config.dimensions])
            
            # Run Batch (Chunked)
            generated_cache = {}   # idx -> trajectory array
            generated_lengths = {} # idx -> real length (int)
            if batch_indices:
                print(f"  [Diffusion] Running batch generation for {len(batch_indices)} samples...")
                ic = {'robot': np.stack(batch_init_cond_robot)}
                if batch_init_cond_obj: ic['obj'] = np.stack(batch_init_cond_obj)
                
                all_goals = np.array(batch_goals)
                all_outs_list = []
                all_lens_list = []
                
                chunk_size = cfg.diffusion.chunk_size
                total_gen = len(batch_indices)
                
                for start_idx in range(0, total_gen, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_gen)
                    
                    chunk_ic = {}
                    if 'robot' in ic: chunk_ic['robot'] = ic['robot'][start_idx:end_idx]
                    if 'obj' in ic: chunk_ic['obj'] = ic['obj'][start_idx:end_idx]
                    
                    chunk_goals = all_goals[start_idx:end_idx]
                    chunk_n = end_idx - start_idx
                    chunk_out, chunk_lens = diff_generator.generate_trajectory(
                        initial_condition=chunk_ic, 
                        goal_condition=chunk_goals, 
                        num_samples=chunk_n,
                        target_traj_length=max_traj_len,
                        use_last_frame_wp=False,
                        enable_goal_stop=False,
                        end_error_threshold=0.25,
                        cfg_w=cfg.diffusion.cfg_weight,
                    )
                    all_outs_list.append(chunk_out)
                    all_lens_list.append(chunk_lens)
                
                # Concatenate all chunks
                outs = np.concatenate(all_outs_list, axis=0) # (Total, T, D)
                lens = np.concatenate(all_lens_list, axis=0) # (Total,)
                generated_cache = {idx: out for idx, out in zip(batch_indices, outs)}
                generated_lengths = {idx: int(l) for idx, l in zip(batch_indices, lens)}
                
                print(f"  [Diffusion] Trajectory lengths: min={lens.min()}, max={lens.max()}, mean={lens.mean():.0f}")

            for i, target_values in enumerate(all_target_values):
                aug_file = iter_dir / f"aug_{i:03d}.npz"
                aug_files.append(aug_file)
                batch_targets.append(target_values)

                # Check if we have a batched diffusion result
                if i in generated_cache:
                     traj = generated_cache[i]
                     t_gen = traj.shape[0]
                     # Real length reported by diffusion (goal-stop / physics-stop)
                     real_len = generated_lengths[i]

                     # Build dict from scratch — avoid copying template's
                     # time-dimensioned arrays at base_traj_len which would
                     # create length mismatches after truncation / padding.
                     template = all_base_data[0]
                     n_bodies = template['body_pos_w'].shape[1]
                     base_T = template['joint_pos'].shape[0]

                     new_data = {}
                     new_data['joint_pos'] = traj[:, 7:36]

                     new_data['body_pos_w'] = np.zeros((t_gen, n_bodies, 3))
                     new_data['body_pos_w'][:,0,:] = traj[:, :3]

                     new_data['body_quat_w'] = np.zeros((t_gen, n_bodies, 4))
                     new_data['body_quat_w'][:,0,:] = traj[:, 3:7]
                     new_data['body_quat_w'][:,1:,0] = 1.0  # Identity for others
                     
                     if traj.shape[1] > 36:
                         new_data['object_pos_w'] = traj[:, 36:39]
                         new_data['object_quat_w'] = traj[:, 39:43]

                     # Copy non-time-dimensioned metadata from template
                     # (e.g. fps, body_names, contact_indices, …)
                     for key, val in template.items():
                         if key in new_data:
                             continue
                         if not isinstance(val, np.ndarray):
                             new_data[key] = val
                         elif val.ndim == 0 or val.shape[0] != base_T:
                             # Scalar array or not time-indexed → safe to copy
                             new_data[key] = val
                     
                     generated_data_batch[i] = new_data
                     traj_lengths.append(real_len)
                     sources.append("diffusion_batch")
                     continue

                # Fallback Logic (non-diffusion sampling methods)
                # Find KNN
                neighbors = elites.get_neighbors(target_values, k=cfg.evolution.neighbor_k, tolerance=cfg.evolution.neighbor_tolerance)
                closest_elite = elites.get_closest(target_values)
                
                if len(neighbors) >= 3 and False: # Disable weighted sum when loading batched initial dataset
                    # Weighted Sum (Exploitation)
                    weighted_sum_trajectory(neighbors, aug_file)
                    new_data = dict(np.load(aug_file))
                    real_len = new_data['joint_pos'].shape[0]
                    if real_len < max_traj_len:
                        for k, v in new_data.items():
                            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == real_len:
                                new_data[k] = np.concatenate([v, np.repeat(v[-1:], max_traj_len - real_len, axis=0)], axis=0)
                    generated_data_batch[i] = new_data
                    traj_lengths.append(real_len)
                    sources.append('weighted_sum')

                elif closest_elite is not None:
                    # Mutation (Exploration)
                    new_data = augment_trajectory_data(closest_elite.data, method='dct')
                    # Use the source elite's traj_length if available, else full array length
                    real_len = closest_elite.traj_length if closest_elite.traj_length > 0 else new_data['joint_pos'].shape[0]
                    arr_len = new_data['joint_pos'].shape[0]
                    if arr_len < max_traj_len:
                        for k, v in new_data.items():
                            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == arr_len:
                                new_data[k] = np.concatenate([v, np.repeat(v[-1:], max_traj_len - arr_len, axis=0)], axis=0)
                    generated_data_batch[i] = new_data
                    traj_lengths.append(real_len)
                    sources.append('dct_mutation')

                else:
                    # DCT Fallback (Initialization)
                    rand_idx = random.randint(0, len(all_base_data)-1)
                    selected_base_data = all_base_data[rand_idx]
                    new_data = augment_trajectory_data(selected_base_data, method='dct')
                    real_len = new_data['joint_pos'].shape[0]
                    if real_len < max_traj_len:
                        for k, v in new_data.items():
                            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == real_len:
                                new_data[k] = np.concatenate([v, np.repeat(v[-1:], max_traj_len - real_len, axis=0)], axis=0)
                    generated_data_batch[i] = new_data
                    traj_lengths.append(real_len)
                    sources.append('dct_base')

            gen_end_time = time.time()
            logger.record_timing("trajectory_generation", gen_end_time - gen_start_time)
            logger.record_sources(sources)
            print(f"[Timer] Trajectory Generation: {gen_end_time - gen_start_time:.2f}s")
            
            # Batch Regenerate Sim Data
            print("Regenerating sim data in batch (in-memory)...")
            
            t_regen = time.time()
            # Regenerate
            if any(sample is None for sample in generated_data_batch):
                raise RuntimeError("Some generated trajectories were not populated before FK regeneration.")
            generated_data_ready = cast(List[Dict[str, Any]], generated_data_batch)
            regenerated_list = cast(
                List[Dict[str, np.ndarray]],
                regenerate_sim_data(generated_data_ready, device=device),
            )
            logger.record_timing("regenerate_sim_data", time.time() - t_regen)
            print(f"[Timer] Regenerate Sim Data: {time.time() - t_regen:.2f}s")
            
            # 2. Batching — stack all trajectories (already padded to max_traj_len)
            print("Batching trajectories...")
            t_batch = time.time()
            batched_data: Dict[str, np.ndarray] = {}
            first_data: Dict[str, np.ndarray] | None = None
            
            # Store traj_lengths array for variable-length masking
            traj_lengths_arr = np.array(traj_lengths, dtype=np.int64)
            
            if regenerated_list:
                first_data = regenerated_list[0]
                keys = list(first_data.keys())
                for key in keys:
                    if isinstance(first_data[key], np.ndarray):
                        if not all(key in d for d in regenerated_list):
                            continue
                        try:
                            batched_data[key] = np.stack([d[key] for d in regenerated_list], axis=0)
                        except Exception:
                            pass  # Skip non-stackable keys (mismatched shapes)
            
            # Add traj_lengths to batched data for downstream use
            if variable_length:
                batched_data['traj_lengths'] = traj_lengths_arr
            
            batched_file = iter_dir / "batched.npz"
            np.savez(batched_file, **cast(Dict[str, Any], batched_data))
            logger.record_timing("batching", time.time() - t_batch)
            print(f"[Timer] Batching: {time.time() - t_batch:.2f}s")
            
            # 3. Run Simulation
            max_steps = max_traj_len if variable_length else (
                first_data["joint_pos"].shape[0]
                if first_data is not None and "joint_pos" in first_data
                else None
            )
            
            sim_start_time = time.time()
            recorded_data = run_batch_simulation_multi_gpu(
                iter_dir=iter_dir,
                batched_file=batched_file,
                task_name=cfg.task,
                num_envs=cfg.num_samples,
                devices=devices,
                start_checkpoint=current_checkpoint,
                viewer=cfg.viewer,
                wandb_run_path=cfg.wandb.run_path,
                max_steps=max_steps,
                batch_targets=batch_targets
            )
            logger.record_timing("simulation", time.time() - sim_start_time)
            print(f"[Timer] Simulation: {time.time() - sim_start_time:.2f}s")

            # Cleanup intermediate files (data is now in memory)
            batched_file.unlink(missing_ok=True)
            (iter_dir / "recorded_positions.npz").unlink(missing_ok=True)

            # 4. Clean Data (Vectorized)
            print("Cleaning batch data...")
            if recorded_data is None:
                raise RuntimeError("Simulation returned no data — check logs above for sim errors.")
            recorded_data = cast(Dict[str, np.ndarray], recorded_data)
            clean_batch_rollouts(recorded_data)
               
            print("Evaluating results (in-memory)...")
                
            # Compute all metrics in memory
            # Use diffusion-generated per-env object Z profile (not base motion)
            ref_obj = batched_data.get('object_pos_w')  # (Envs, T, 3)
            if ref_obj is not None:
                ref_obj = np.transpose(ref_obj, (1, 0, 2))  # -> (T, Envs, 3)
            batch_results = compute_batch_metrics(
                recorded_data, cfg.num_samples, cfg.task,
                evo_task_name=evo_task_name,
                base_object_pos_w=ref_obj,
                traj_lengths=traj_lengths_arr if variable_length else None,
            )
            
            # --- Vectorized elite selection ---
            fitness_arr = np.array([res["fitness"] for res in batch_results])
            elite_mask = fitness_arr > cfg.evolution.fitness_threshold
            elite_indices = np.where(elite_mask)[0]

            logger.record_fitness(fitness_arr)
            logger.num_passed_fitness = len(elite_indices)
            
            print(f"  {len(elite_indices)}/{cfg.num_samples} samples passed fitness threshold.")
            
            # Batch merge recorded data for all passing elites at once
            t_merge = time.time()
            if len(elite_indices) > 0:
                batched_elite_data = overwrite_with_rollouts(
                    regenerated_list, recorded_data, elite_indices
                )
                # Save single batched file for this iteration's elites
                iter_elites_file = iter_dir / "iter_elites.npz"
                np.savez(iter_elites_file, **cast(Dict[str, Any], batched_elite_data))
            else:
                batched_elite_data = {}
            logger.record_timing("elite_merge_save", time.time() - t_merge)
            print(f"[Timer] Batch elite merge + save: {time.time() - t_merge:.2f}s")
            
            # Build Elite objects and update archive
            new_elites_count = 0
            new_elites_added = []
            elite_accepted_sources = []
            outcomes = ["rejected"] * cfg.num_samples
            
            for batch_pos, env_i in enumerate(elite_indices):
                res = batch_results[env_i]
                
                # Extract individual data dict from the batched arrays
                elite_data_single = {}
                for k, v in batched_elite_data.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > batch_pos:
                        elite_data_single[k] = v[batch_pos]
                    else:
                        elite_data_single[k] = v
                
                new_elite = Elite(
                    task_metrics=res["task_metrics"],
                    fitness=res["fitness"],
                    data=elite_data_single,
                    source=sources[env_i],
                    metrics=res["metrics"],
                    iteration=iteration,
                    sample_idx=env_i,
                    generation=generation,
                    file=str(out_root),
                    batch_index=int(batch_pos),
                    traj_length=int(traj_lengths[env_i]) if variable_length else -1,
                )
                
                status = elites.add(new_elite)
                outcomes[env_i] = status
                if status == "new":
                    logger.new_bins_filled += 1
                    new_elites_count += 1
                    new_elites_added.append(new_elite)
                    elite_accepted_sources.append(sources[env_i])
                elif status == "replaced":
                    logger.old_bins_replaced += 1
                    new_elites_count += 1
                    new_elites_added.append(new_elite)
                    elite_accepted_sources.append(sources[env_i])
                else:
                    logger.bins_rejected += 1

            if cfg.curriculum == "thompson":
                elites.update_thompson_statistics(
                    sampled_bins=sampled_bin_tuples,
                    outcomes=outcomes,
                    fitnesses=fitness_arr.tolist(),
                )

            logger.record_elite_sources(elite_accepted_sources)
            print(f"  Found {new_elites_count} new elites in this iteration.")
            new_elites_count_cumulative += new_elites_count

            # --- Plotting Targets vs Achieved (New Elites) ---
            if len(task_config.dimensions) >= 6:
                plot_targets_vs_achieved_6d(
                    task_config=task_config,
                    targets=batch_targets,
                    new_elites=new_elites_added,
                    generation=generation,
                    iteration=iteration,
                    output_dir=iter_dir,
                )
            else:
                plot_targets_vs_achieved(
                    task_config=task_config,
                    targets=batch_targets,
                    new_elites=new_elites_added,
                    generation=generation,
                    iteration=iteration,
                    output_dir=iter_dir
                )

            # 6. Train Diffusion Model
            if len(elites) > 0:
                _should_train = False

                if not cfg.diffusion.reset_every:
                    # --- Mode A: no periodic reset → train every iteration ---
                    diff_subset = elites.downsample(
                        method=cfg.diffusion.sampling,
                        budget=cfg.diffusion.subset_size,
                        num_bins=cfg.diffusion.num_bins,
                    )
                    _should_train = True
                    is_diff_reset = False
                    print(f"  [Diffusion] Resampled {cfg.diffusion.sampling} subset ({len(diff_subset)} elites) [every-iter mode]")
                else:
                    # --- Mode B: periodic reset → train only on reset generations ---
                    is_diff_reset = (generation % cfg.diffusion.reset_every == 0
                                     and iteration == 0)
                    if diff_subset is None or is_diff_reset:
                        diff_subset = elites.downsample(
                            method=cfg.diffusion.sampling,
                            budget=cfg.diffusion.subset_size,
                            num_bins=cfg.diffusion.num_bins,
                        )
                        _should_train = True
                        print(f"  [Diffusion] Resampled {cfg.diffusion.sampling} subset ({len(diff_subset)} elites) [reset-every mode]")
                    else:
                        print(f"  [Diffusion] Skipping (reusing subset from iteration 0)")

                if _should_train:
                    training_subset = diff_subset
                    elite_data = [e.data for e in training_subset]

                    elite_targets = []
                    for e in training_subset:
                         vec = [float(e.task_metrics[dim.name]) for dim in task_config.dimensions]
                         elite_targets.append(vec)

                    # Plot which elites were selected for training
                    plot_diffusion_training_subset(
                        task_config=task_config,
                        archive=elites,
                        training_subset=training_subset,
                        output_path=iter_dir / "diffusion_training_subset.png",
                    )

                    print(f"  [Diffusion Training] Training on {len(elite_data)}/{len(elites)} elites ({cfg.diffusion.sampling})...")
                    if new_elites_count_cumulative > cfg.diffusion.stats_elite_threshold:
                        num_epochs = cfg.diffusion.train_epochs * 2 # More epochs when we have more new data
                    else:
                        num_epochs = cfg.diffusion.train_epochs

                    # Normalization stats are always reused from the initial base-data
                    # fit() call.  Recomputing on elite data would cause distribution
                    # drift and degrade generation quality.
                    calc_stats = False

                    # Periodic from-scratch reset to prevent accumulated bias
                    resume_ckpt = last_ckpt_path
                    if is_diff_reset:
                        print(f"  [Diffusion Training] *** Resetting model from scratch (every {cfg.diffusion.reset_every} gens) ***")
                        resume_ckpt = None
                        calc_stats = False
                        num_epochs = cfg.diffusion.reset_epochs

                    t_diff_train = time.time()
                    last_ckpt_path = diff_generator.fit(
                        data_source=elite_data,
                        epochs=num_epochs,
                        save_path=str(gen_dir / "diffusion_gen.pt"),
                        checkpoint=resume_ckpt,
                        calc_stats=calc_stats,
                    )
                    logger.record_timing("diffusion_training", time.time() - t_diff_train)
                    print(f"  [Diffusion Training] Done. Checkpoint: {last_ckpt_path}")

                    del elite_data, training_subset

            # --- Memory cleanup between iterations ---
            del generated_data_batch, regenerated_list, batched_data, generated_cache
            del batch_init_cond_robot, batch_init_cond_obj, batch_goals, batch_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory(f"iter-{iteration}-cleanup")

            # CLEANUP: Delete individual aug files (elites are now in iter_elites.npz)
            for i in range(cfg.num_samples):
                f_path = aug_files[i]
                if f_path.exists():
                    try:
                        f_path.unlink()
                    except Exception:
                        pass
                
            print(f"  Total Elites: {len(elites)}")

            # --- Archive coverage plot (6D) ---
            if len(task_config.dimensions) >= 6 and len(elites) > 0:
                plot_archive_6d(
                    task_config=task_config,
                    archive=elites,
                    output_path=iter_dir / "archive_coverage_6d.png",
                    title=f"Archive Gen {generation} Iter {iteration} ({len(elites)} elites)",
                )

            # --- Iteration Summary ---
            logger.archive_size_after = len(elites)
            logger.finalize()
            logger.print_summary()
            logger.save_json(iter_dir / "iteration_summary.json")
            logger.log_wandb()
            
            # Save Elite Summary
            elites.save(gen_dir / "elites.json")
        
        # 7. Train Policy on Elites

        if cfg.train:
            print(f"\n=== Training Policy on Elites (Gen {generation}) ===")
            
            motion_file_to_use = None
   
            if len(elites) > 0:
                elites_file = gen_dir / "elites.npz"
                
                # Base trajectories are already in the archive (seeded at startup)
                training_elites = list(elites.values())
                
                if batch_elites(training_elites, elites_file):
                    motion_file_to_use = str(elites_file)
            
            if motion_file_to_use:
                new_checkpoint = run_policy_training(
                    task_name=cfg.task,
                    motion_file=motion_file_to_use,
                    device=device,
                    resume_checkpoint=current_checkpoint,
                    num_envs=cfg.training.num_envs,
                    max_iterations=cfg.training.elite_max_iterations,
                    wandb_project=cfg.wandb.project,
                    num_gpus=effective_num_gpus,
                )
                
                if new_checkpoint:
                    current_checkpoint = new_checkpoint
                    print(f"[INFO] Updating checkpoint for next generation: {current_checkpoint}")
                else:
                    print(f"[WARN] Failed to update checkpoint.")


        # Finish WandB run at the end of the generation
        if wandb.run:
            wandb.finish()

      except Exception as e:
        print(f"\n[FATAL] Generation {generation} failed with error: {e}")
        traceback.print_exc()
        if wandb.run:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main()
