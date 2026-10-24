import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, cast
from humanoid_dart.scripts.play import run_play, PlayConfig, HeadlessRecordingViewer
from humanoid_dart.scripts.train import run_train, TrainConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from humanoid_dart.tracking.mdp.commands import MotionLoader
from humanoid_dart.dart.multi_device import split_list, split_array, merge_array_dicts

def find_latest_checkpoint(log_root_path: Path) -> Optional[str]:
    """Finds the latest model checkpoint in a given log directory."""
    try:
        if not log_root_path.exists():
             return None
             
        subdirs = [d for d in log_root_path.iterdir() if d.is_dir()]
        if not subdirs:
            return None
            
        latest_run_dir = max(subdirs, key=os.path.getmtime)
        model_files = list(latest_run_dir.glob("model_*.pt"))
        if not model_files:
            return None
            
        latest_model = max(model_files, key=lambda p: int(p.stem.split('_')[1]))
        return str(latest_model)
    except Exception as e:
        print(f"[Warn] Failed to find checkpoint: {e}")
        return None

def run_policy_training(
    task_name: str,
    motion_file: str,
    device: str,
    resume_checkpoint: Optional[str] = None,
    num_envs: int = 2048,
    max_iterations: int = 1200,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    num_gpus: int = 1,
) -> Optional[str]:
    """
    Runs PPO training for the policy.

    When *num_gpus* > 1, training is launched as a ``torch.distributed.run`` subprocess
    with ``--distributed True`` so each GPU runs its own env batch and
    gradients are synchronised via ``all_reduce``.
    
    Returns:
        Path to the new latest checkpoint if training succeeded, else None.
    """
    # Load default configs (needed for experiment_name / checkpoint path)
    env_cfg = load_env_cfg(task_name)
    agent_cfg = load_rl_cfg(task_name)

    if num_gpus > 1:
        return _run_policy_training_distributed(
            task_name=task_name,
            motion_file=motion_file,
            resume_checkpoint=resume_checkpoint,
            num_envs=num_envs,
            max_iterations=max_iterations,
            wandb_project=wandb_project,
            run_name=run_name,
            num_gpus=num_gpus,
            experiment_name=agent_cfg.experiment_name,
        )

    print(f"Starting training on {motion_file}...")
    
    checkpoint_file = str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None
    if checkpoint_file:
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_file}")
    
    # Override configs
    env_cfg.scene.num_envs = num_envs
    agent_cfg.max_iterations = max_iterations
    
    # Configure WandB (TensorBoard when no project is set)
    if wandb_project:
        agent_cfg.logger = "wandb"
        agent_cfg.wandb_project = wandb_project
    else:
        agent_cfg.logger = "tensorboard"
    if run_name:
        agent_cfg.run_name = run_name

    # Create TrainConfig
    train_cfg = TrainConfig(
        env=env_cfg,
        agent=agent_cfg,
        motion_file=motion_file,
        checkpoint_file=checkpoint_file,
        device=device
    )

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[CUDA] free {free/1e9:.2f} GB / total {total/1e9:.2f} GB")
    
    # Run Training
    run_train(task_name, train_cfg)
    
    # Find new checkpoint
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    return find_latest_checkpoint(log_root_path)


def _run_policy_training_distributed(
    task_name: str,
    motion_file: str,
    resume_checkpoint: Optional[str],
    num_envs: int,
    max_iterations: int,
    wandb_project: Optional[str],
    run_name: Optional[str],
    num_gpus: int,
    experiment_name: str,
) -> Optional[str]:
    """Launch PPO training via ``torchrun`` for multi-GPU gradient sync."""
    import subprocess

    if num_gpus < 2:
        raise ValueError(f"Distributed training requires at least 2 GPUs, got {num_gpus}")

    visible_cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_cuda_devices < num_gpus:
        raise ValueError(
            f"Requested {num_gpus} GPUs for distributed training, but only "
            f"{visible_cuda_devices} CUDA devices are visible"
        )

    print(f"[Multi-GPU] Launching distributed PPO training on {num_gpus} GPUs...")

    # Run the train module with the current interpreter, so the subprocess
    # uses the same environment as the pipeline.
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "-m", "humanoid_dart.scripts.train", task_name,
        "--distributed", "True",
        "--motion-file", str(Path(motion_file).resolve()),
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(max_iterations),
    ]

    if resume_checkpoint:
        cmd += ["--checkpoint-file", str(Path(resume_checkpoint).resolve())]

    if wandb_project:
        cmd += ["--agent.logger", "wandb", "--agent.wandb-project", wandb_project]
    else:
        cmd += ["--agent.logger", "tensorboard"]
    if run_name:
        cmd += ["--agent.run-name", run_name]

    print(f"[Multi-GPU] CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"[Multi-GPU] ERROR: distributed training exited with code {result.returncode}")
        return None

    log_root_path = (Path("logs") / "rsl_rl" / experiment_name).resolve()
    return find_latest_checkpoint(log_root_path)

def run_batch_simulation(
    iter_dir: Path,
    batched_file: Path,
    task_name: str,
    num_envs: int,
    start_checkpoint: Optional[str] = None,
    device: str = "cuda:0",
    viewer: str = "auto",
    wandb_run_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    batch_targets: Optional[List[Dict[str, float]]] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Runs the simulation for a batch of trajectories.
    
    Args:
        iter_dir: Directory to run the simulation in (where output files will be saved).
        batched_file: Path to the .npz file containing batched trajectories.
        task_name: Name of the RL task.
        num_envs: Number of environments to run.
        start_checkpoint: Path to the RL policy checkpoint.
        device: Device to run on.
        viewer: Viewer mode ("auto", "native", "viser", "headless").
        wandb_run_path: Optional WandB run path for loading artifacts.
        max_steps: Maximum number of simulation steps.
        batch_targets: Optional target values for the batch (used in tracking tasks).
        
    Returns:
        Loaded recorded data as numpy array (or dict of arrays) if successful, None otherwise.
        The data is loaded from 'recorded_positions.npz' generated by the simulation.
    """
    
    print("Running simulation...")
    
    # Resolve all paths to absolute before calling run_play
    checkpoint_file_abs = str(Path(start_checkpoint).resolve()) if start_checkpoint else None
    batched_file_abs = str(Path(batched_file).resolve())
    iter_dir_abs = str(Path(iter_dir).resolve())
    
    try:
        cfg_kwargs: dict = {}
        if wandb_run_path:
            cfg_kwargs["wandb_run_path"] = wandb_run_path
        if checkpoint_file_abs:
            cfg_kwargs["checkpoint_file"] = checkpoint_file_abs

        cfg_kwargs["motion_file"] = batched_file_abs
        cfg_kwargs["num_envs"] = num_envs
        cfg_kwargs["device"] = device
        
        if max_steps is not None:
            cfg_kwargs["max_steps"] = max_steps

        cfg_kwargs["viewer"] = viewer
        cfg_kwargs["output_dir"] = iter_dir_abs
        cfg_kwargs["skip_onnx_export"] = True

        cfg = PlayConfig(**cfg_kwargs)
        
        # Run play
        run_play(task_name, cfg, batch_target_values=batch_targets)
        
        # Check for output file
        recorded_name = Path(iter_dir_abs) / "recorded_positions.npz"
        
        # Simple retry logic for file system latency
        if not recorded_name.exists():
            time.sleep(2.0)
        
        if not recorded_name.exists():
            print(f"  [Error] {recorded_name} not found!")
            return None
            
        print("Loading simulation results...")
        with np.load(recorded_name) as npz:
            recorded_data = {k: npz[k] for k in npz.files}
        recorded_name.unlink(missing_ok=True)
        return recorded_data

    except Exception as e:
        print(f"[Error] Simulation run failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_batch_simulation_multi_gpu(
    iter_dir: Path,
    batched_file: Path,
    task_name: str,
    num_envs: int,
    devices: list[str],
    start_checkpoint: Optional[str] = None,
    viewer: str = "auto",
    wandb_run_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    batch_targets: Optional[List[Dict[str, float]]] = None,
) -> Optional[dict[str, np.ndarray]]:
    """Run simulation split across multiple GPUs **in parallel**.

    Each GPU chunk is launched as a separate subprocess (via
    ``rollout_worker.py``) so that every process gets its own Warp/CUDA
    context — no shared-stream conflicts.

    When ``len(devices) == 1`` this falls back to the in-process
    single-device ``run_batch_simulation``.
    """
    import json
    import shutil
    import subprocess

    n = len(devices)
    if n <= 1:
        return run_batch_simulation(
            iter_dir=iter_dir, batched_file=batched_file,
            task_name=task_name, num_envs=num_envs,
            start_checkpoint=start_checkpoint, device=devices[0],
            viewer=viewer, wandb_run_path=wandb_run_path,
            max_steps=max_steps, batch_targets=batch_targets,
        )

    # Load the full batched data and split into per-device chunks
    with np.load(batched_file) as npz:
        full_data = {k: npz[k] for k in npz.files}

    batch_dim = num_envs
    chunk_sizes = []
    k_quo, k_rem = divmod(batch_dim, n)
    for i in range(n):
        chunk_sizes.append(k_quo + (1 if i < k_rem else 0))

    # Resolve checkpoint once
    ckpt_abs = str(Path(start_checkpoint).resolve()) if start_checkpoint else None

    # --- Prepare per-device work and launch subprocesses ---
    scratch_base = Path(os.environ.get("TMPDIR", "/tmp"))
    scratch_root = scratch_base / f"mjlab_folm_rollout_{os.getpid()}_{int(time.time())}"
    scratch_root.mkdir(parents=True, exist_ok=True)
    print(f"  [Multi-GPU] Using scratch dir: {scratch_root}")

    processes: list[tuple[int, subprocess.Popen, Path, Any]] = []
    offset = 0

    for dev_idx, (dev, csize) in enumerate(zip(devices, chunk_sizes)):
        if csize == 0:
            continue

        sub_dir = scratch_root / f"gpu_{dev_idx}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Slice arrays along batch dimension
        chunk_data: dict[str, np.ndarray] = {}
        for key, arr in full_data.items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] == batch_dim:
                chunk_data[key] = arr[offset:offset + csize]
            else:
                chunk_data[key] = arr
        sub_file = sub_dir / "batched.npz"
        np.savez(sub_file, **chunk_data)

        # Serialize batch_targets to JSON for the subprocess
        targets_file = None
        if batch_targets:
            sub_targets = batch_targets[offset:offset + csize]
            targets_path = sub_dir / "targets.json"
            with open(targets_path, "w") as f:
                json.dump(sub_targets, f)
            targets_file = str(targets_path.resolve())

        cmd = [
            sys.executable, "-m", "humanoid_dart.dart.rollout_worker",
            "--task", task_name,
            "--motion-file", str(sub_file.resolve()),
            "--output-dir", str(sub_dir.resolve()),
            "--num-envs", str(csize),
            "--device", dev,
            "--viewer", viewer,
        ]
        if ckpt_abs:
            cmd += ["--checkpoint", ckpt_abs]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        if targets_file:
            cmd += ["--targets-file", targets_file]

        print(f"  [Multi-GPU] Launching {csize} envs on {dev} (subprocess)")
        log_file = sub_dir / "worker.log"
        log_fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=log_fh, stderr=log_fh)
        processes.append((dev_idx, proc, sub_dir, log_fh))
        offset += csize

    # --- Wait for all subprocesses to complete ---
    print(f"  [Multi-GPU] Waiting for {len(processes)} rollout workers...")
    per_device_results: list[dict[str, np.ndarray]] = []
    failed = False

    for dev_idx, proc, sub_dir, log_fh in processes:
        rc = proc.wait()
        log_fh.close()
        log_file = sub_dir / "worker.log"
        if rc != 0:
            print(f"  [Multi-GPU] ERROR: GPU {dev_idx} worker exited with code {rc}")
            if log_file.exists():
                print(f"  [Multi-GPU] --- Worker log (GPU {dev_idx}) ---")
                print(log_file.read_text())
                print(f"  [Multi-GPU] --- End worker log (GPU {dev_idx}) ---")
            failed = True
            continue

        recorded_file = sub_dir / "recorded_positions.npz"
        if not recorded_file.exists():
            import time as _time
            _time.sleep(2.0)
        if not recorded_file.exists():
            print(f"  [Multi-GPU] ERROR: No output from GPU {dev_idx}")
            if log_file.exists():
                print(f"  [Multi-GPU] --- Worker log (GPU {dev_idx}) ---")
                print(log_file.read_text())
                print(f"  [Multi-GPU] --- End worker log (GPU {dev_idx}) ---")
            failed = True
            continue

        with np.load(recorded_file) as npz:
            per_device_results.append({k: npz[k] for k in npz.files})

        # Clean up sub-files
        recorded_file.unlink(missing_ok=True)
        (sub_dir / "batched.npz").unlink(missing_ok=True)
        (sub_dir / "targets.json").unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)

    if failed:
        # Kill any remaining workers
        for _, proc, _, log_fh in processes:
            try:
                log_fh.close()
            except Exception:
                pass
            proc.kill()
        shutil.rmtree(scratch_root, ignore_errors=True)
        return None

    print(f"  [Multi-GPU] All {len(processes)} workers completed successfully")

    # Recorded data is time-first: (T, Envs, ...) for 2D+ arrays,
    # but (Envs,) for 1D arrays like traj_lengths.
    # merge_array_dicts concatenates on axis 0, which is wrong for
    # time-first arrays — we need to concat on axis 1 (the env axis).
    merged: dict[str, np.ndarray] = {}
    for key in per_device_results[0]:
        vals = [d[key] for d in per_device_results if key in d]
        if isinstance(vals[0], np.ndarray):
            if vals[0].ndim >= 2:
                # Time-first layout: concat along env axis (axis 1)
                merged[key] = np.concatenate(vals, axis=1)
            elif vals[0].ndim == 1:
                # 1D = per-env data: concat along axis 0
                merged[key] = np.concatenate(vals, axis=0)
            else:
                merged[key] = vals[0]
        else:
            merged[key] = vals[0]

    shutil.rmtree(scratch_root, ignore_errors=True)
    return merged
