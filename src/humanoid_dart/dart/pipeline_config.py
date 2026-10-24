"""Configuration dataclass for the DART evolutionary pipeline.

Values can be set via a YAML config file (``--config``), CLI flags, or both.
CLI flags always override config-file values, which override defaults.

Example YAML::

    task: Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation
    base_motion: motions/pick_place.npz
    generations: 5
    training:
      num_envs: 4096
      initial_max_iterations: 2000
    diffusion:
      subset_size: 150
      reset_every: 2
"""

from __future__ import annotations

import argparse
import yaml
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Nested sub-configs
# ---------------------------------------------------------------------------

@dataclass
class WandbConfig:
    """Weights & Biases logging settings (disabled when project is None)."""
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    run_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """RL policy training hyperparameters."""
    num_envs: int = 8192
    initial_max_iterations: int = 4000
    elite_max_iterations: int = 1200


@dataclass
class DiffusionConfig:
    """Diffusion model hyperparameters."""
    config_path: Optional[str] = None
    checkpoint: Optional[str] = None
    subset_size: int = 200
    reset_every: int = 1
    sampling: str = "frontier"
    num_bins: int = 4
    initial_epochs: int = 5
    train_epochs: int = 5
    reset_epochs: int = 40
    chunk_size: int = 3000
    stats_elite_threshold: int = 50
    cfg_weight: float = 1.0

@dataclass
class ReplanningConfig:
    """Closed-loop replanning settings."""
    enabled: bool = False
    replan_freq: int = 50
    horizon: int = 0
    start_generation: int = 0


@dataclass
class EvolutionConfig:
    """Evolution loop hyperparameters."""
    fitness_threshold: float = 0.0005
    diffusion_probability: float = 1.0
    neighbor_k: int = 5
    neighbor_tolerance: float = 0.02
    min_traj_length: Optional[int] = None
    max_traj_length: Optional[int] = None


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Full configuration for the evolutionary pipeline."""

    # --- Required (no defaults) ---
    task: str = ""
    base_motion: str = ""

    # --- General ---
    evo_task: Optional[str] = None
    out_dir: str = "evo_results"
    device: str = "cuda:0"
    num_gpus: int = 1
    viewer: str = "auto"
    num_samples: int = 10
    iterations: int = 5
    generations: int = 1
    train: bool = False
    curriculum: str = "random"
    checkpoint_file: Optional[str] = None
    load_elites: Optional[str] = None

    # --- Nested ---
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    replanning: ReplanningConfig = field(default_factory=ReplanningConfig)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "PipelineConfig":
        nested_map = {
            "wandb": WandbConfig,
            "training": TrainingConfig,
            "diffusion": DiffusionConfig,
            "evolution": EvolutionConfig,
            "replanning": ReplanningConfig,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in nested_map and isinstance(value, dict):
                kwargs[key] = nested_map[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# CLI argument parsing  (config file + overrides)
# ---------------------------------------------------------------------------

def parse_pipeline_args(argv: list[str] | None = None) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from an optional YAML file + CLI flags.

    Precedence: CLI flag  >  config file  >  dataclass default.
    """
    parser = argparse.ArgumentParser(
        description="Evolutionary pipeline for trajectory generation.",
    )

    # Config file ---------------------------------------------------------
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. CLI flags override values in the file.",
    )

    # Top-level -----------------------------------------------------------
    parser.add_argument("--task", type=str, default=None, help="RL task name")
    parser.add_argument("--evo_task", type=str, default=None, help="Evolutionary task config name")
    parser.add_argument("--base_motion", type=str, default=None, help="Path to base motion")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use (1-4)")
    parser.add_argument("--viewer", type=str, default=None, choices=["auto", "native", "viser", "headless"])
    parser.add_argument("--num_samples", "--num", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--train", action="store_true", default=None)
    parser.add_argument("--curriculum", type=str, default=None, choices=["random", "frontier", "bayesian", "thompson"])
    parser.add_argument("--checkpoint_file", type=str, default=None)
    parser.add_argument("--load_elites", type=str, default=None)

    # WandB ---------------------------------------------------------------
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_run_path", type=str, default=None)

    # Training ------------------------------------------------------------
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--initial_max_iterations", type=int, default=None)
    parser.add_argument("--elite_max_iterations", type=int, default=None)

    # Diffusion -----------------------------------------------------------
    parser.add_argument("--diff_config", type=str, default=None)
    parser.add_argument("--diff_checkpoint", type=str, default=None)
    parser.add_argument("--diff_subset_size", type=int, default=None)
    parser.add_argument("--diff_reset_every", type=int, default=None)
    parser.add_argument("--diff_sampling", type=str, default=None, choices=["stratified", "clustered", "frontier"])
    parser.add_argument("--diff_num_bins", type=int, default=None)
    parser.add_argument("--diff_initial_epochs", type=int, default=None)
    parser.add_argument("--diff_train_epochs", type=int, default=None)
    parser.add_argument("--diff_reset_epochs", type=int, default=None)
    parser.add_argument("--diff_chunk_size", type=int, default=None)
    parser.add_argument("--diff_stats_elite_threshold", type=int, default=None)
    parser.add_argument("--diff_cfg_weight", type=float, default=None)

    # Evolution -----------------------------------------------------------
    parser.add_argument("--fitness_threshold", type=float, default=None)
    parser.add_argument("--diffusion_probability", type=float, default=None)
    parser.add_argument("--neighbor_k", type=int, default=None)
    parser.add_argument("--neighbor_tolerance", type=float, default=None)
    parser.add_argument("--min_traj_length", type=int, default=None)
    parser.add_argument("--max_traj_length", type=int, default=None)

    # Replanning ----------------------------------------------------------
    parser.add_argument("--replanning", action="store_true", default=None, help="Enable closed-loop replanning during simulation")
    parser.add_argument("--replan_freq", type=int, default=None, help="Steps between replanning")
    parser.add_argument("--replan_horizon", type=int, default=None, help="Diffusion trajectory length for replanning (0=use default)")
    parser.add_argument("--replan_start_generation", type=int, default=None, help="Generation to start replanning")

    args = parser.parse_args(argv)

    # 1. Start from defaults (or YAML if provided)
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig()

    # 2. Override top-level fields from CLI
    _cli = vars(args)
    _top_fields = {f.name for f in fields(PipelineConfig) if f.name not in ("wandb", "training", "diffusion", "evolution", "replanning")}
    for key in _top_fields:
        cli_val = _cli.get(key)
        if cli_val is not None:
            setattr(cfg, key, cli_val)

    # 3. Override nested WandB fields
    _wandb_map = {
        "wandb_project": "project",
        "wandb_entity": "entity",
        "wandb_name": "name",
        "wandb_run_path": "run_path",
    }
    for cli_key, attr in _wandb_map.items():
        val = _cli.get(cli_key)
        if val is not None:
            setattr(cfg.wandb, attr, val)

    # 4. Override nested Training fields
    _train_map = {
        "num_envs": "num_envs",
        "initial_max_iterations": "initial_max_iterations",
        "elite_max_iterations": "elite_max_iterations",
    }
    for cli_key, attr in _train_map.items():
        val = _cli.get(cli_key)
        if val is not None:
            setattr(cfg.training, attr, val)

    # 5. Override nested Diffusion fields
    _diff_map = {
        "diff_config": "config_path",
        "diff_checkpoint": "checkpoint",
        "diff_subset_size": "subset_size",
        "diff_reset_every": "reset_every",
        "diff_sampling": "sampling",
        "diff_num_bins": "num_bins",
        "diff_initial_epochs": "initial_epochs",
        "diff_train_epochs": "train_epochs",
        "diff_reset_epochs": "reset_epochs",
        "diff_chunk_size": "chunk_size",
        "diff_stats_elite_threshold": "stats_elite_threshold",
        "diff_cfg_weight": "cfg_weight",
    }
    for cli_key, attr in _diff_map.items():
        val = _cli.get(cli_key)
        if val is not None:
            setattr(cfg.diffusion, attr, val)

    # 6. Override nested Evolution fields
    _evo_map = {
        "fitness_threshold": "fitness_threshold",
        "diffusion_probability": "diffusion_probability",
        "neighbor_k": "neighbor_k",
        "neighbor_tolerance": "neighbor_tolerance",
        "min_traj_length": "min_traj_length",
        "max_traj_length": "max_traj_length",
    }
    for cli_key, attr in _evo_map.items():
        val = _cli.get(cli_key)
        if val is not None:
            setattr(cfg.evolution, attr, val)

    # 7. Override nested Replanning fields
    _replan_map = {
        "replanning": "enabled",
        "replan_freq": "replan_freq",
        "replan_horizon": "horizon",
        "replan_start_generation": "start_generation",
    }
    for cli_key, attr in _replan_map.items():
        val = _cli.get(cli_key)
        if val is not None:
            setattr(cfg.replanning, attr, val)

    # Validate required fields
    if not cfg.task:
        parser.error("--task is required (via CLI or config file)")
    if not cfg.base_motion:
        parser.error("--base_motion is required (via CLI or config file)")

    return cfg
