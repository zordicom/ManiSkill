"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

from typing import List, Tuple

import attr

from zordi_vla.configs.configs import BaseModelConfig, DatasetConfig


@attr.s(auto_attribs=True, frozen=True)
class NetworkConfig(BaseModelConfig):
    """Configuration for network architectures."""

    # State encoder dimensions
    state_encoder_dims: List[int] = [128, 64]

    # Expert action encoder dimensions
    expert_action_encoder_dims: List[int] = [64, 32]

    # Image encoder CNN parameters
    image_conv_channels: List[int] = [32, 64, 64]
    image_conv_kernel_sizes: List[int] = [8, 4, 3]
    image_conv_strides: List[int] = [4, 2, 1]
    image_adaptive_pool_size: Tuple[int, int] = (4, 4)
    image_fc_dim: int = 128

    # Fusion layer dimensions
    fusion_dims: List[int] = [256, 128]

    # Policy head dimensions
    policy_head_dims: List[int] = [128, 64]

    # Value head dimensions
    value_head_dims: List[int] = [128, 64]

    # Policy log_std initial value
    log_std_init: float = -0.5

    # Whether policy and value networks share a single encoder instance
    use_shared_encoder: bool = False

    # Extra observation processing settings
    extra_obs_encoder_dim: int = 32  # Hidden dimension for extra obs encoders
    extra_obs_output_dim: int = 16  # Output dimension for each extra obs field


@attr.s(auto_attribs=True, frozen=True)
class RolloutBufferConfig:
    """Configuration for rollout buffer management and sampling."""

    # Two-bucket sampling configuration
    demo_fraction: float = 0.3  # Fraction of samples from bucket A (base policy demos)

    # Grade-weighted sampling for bucket B (residual rollouts)
    grade_weight_beta: float = 0.5  # Exponential weight for grade levels (higher = more bias toward high grades)
    min_grade: int = 2  # Minimum grade level to include in bucket B

    # Age-based decay for bucket B
    age_decay_lambda: float = (
        0.1  # Exponential decay rate for age (higher = more bias toward recent)
    )
    max_age_steps: int = 10  # Maximum age steps to include in bucket B

    # Buffer size management
    max_episodes_per_model: int = 50  # Maximum episodes to keep per model directory
    max_total_episodes: int = 500  # Maximum total episodes across all models

    # Validation holdout
    val_holdout_fraction: float = (
        0.05  # Fraction of best episodes to hold out for validation
    )


@attr.s(auto_attribs=True, frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm parameters."""

    # Core PPO hyperparameters
    gamma: float = 0.99
    clip_ratio: float = 0.1
    train_epochs: int = 5
    mini_batch_size: int = 64

    # Training parameters (separate learning rates for policy and value nets)
    policy_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    entropy_bonus: float = 0.01
    grad_clip_norm: float = 0.5
    target_kl: float = 0.015

    # Delta action penalty coefficient (||delta||^2 regulariser)
    delta_penalty_coeff: float = 0.01


@attr.s(auto_attribs=True, frozen=True)
class SACConfig:
    """Configuration for SAC algorithm parameters."""

    # Core SAC hyperparameters
    gamma: float = 0.99
    polyak: float = 0.005

    # UTD (Update-to-Data) ratio control
    # Number of gradient updates per outer training epoch
    # Higher values = more aggressive learning from same data (higher UTD)
    # For UTD=1: set this to (env_steps_collected / training.num_epochs)
    gradient_updates_per_epoch: int = 5

    # Mini-batch size for gradient updates (used in both update methods)
    # This is the actual batch size seen by the networks during each update
    mini_batch_size: int = 64

    # Training parameters (separate learning rates for policy and critics)
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    grad_clip_norm: float = 0.5

    # Automatic temperature tuning (optimized for delta actions)
    auto_temp: bool = True
    target_entropy: float | None = (
        None  # If None, uses -0.1 * action_dim for delta actions
    )

    # Delta action output range control
    delta_action_max_range: float = 0.05  # Maximum magnitude for delta actions (±range)

    # Delta action penalty coefficient with scheduling (||delta||^2 regulariser)
    delta_penalty_coeff: float = 0.0  # Start with 0, can be scheduled
    delta_penalty_schedule: bool = True  # Whether to schedule delta penalty
    delta_penalty_max: float = 0.01  # Maximum delta penalty after scheduling
    delta_penalty_epochs: int = 100  # Epochs to reach max penalty

    # Conservative Q-Learning (CQL) for offline RL stability
    cql_weight: float = 1.0  # Conservative penalty weight
    cql_temp: float = 1.0  # Temperature for CQL log-sum-exp
    cql_hinge: bool = False  # Whether to use hinge loss variant
    cql_use_lagrange: bool = False  # Adaptive CQL weight
    cql_target: float = 0.0  # Target value for adaptive CQL
    cql_lambda_lr: float = 1e-3  # Learning rate for adaptive CQL weight

    # Modern SAC improvements
    bc_weight: float = 0.0  # Behavior cloning regularization weight
    action_reg_weight: float = 0.0  # L2 regularization on action magnitude
    anchor_weight: float = 0.0  # Anchor loss to batch reward mean
    use_value_norm: bool = False  # Value normalization for Q-targets

    # Performance optimization settings
    efficient_cql: bool = True  # Use batched CQL computation to reduce forward passes


@attr.s(auto_attribs=True, frozen=True)
class TrainingConfig:
    """Configuration for training loop parameters."""

    # Training loop settings
    num_epochs: int = 100
    # DataLoader batch size - how many samples to load from dataset per DataLoader iteration
    # This affects memory usage and data loading efficiency, not gradient computation
    dataloader_batch_size: int = 128
    num_workers: int = 4
    use_gpu: bool = True

    # Controlled exploration std used when collecting new rollouts
    exploration_std: float = 0.02

    # Resume functionality
    resume_checkpoint_path: str | None = None  # Path to checkpoint to resume from

    # Logging settings
    log_interval: int = 10
    val_interval: int = 20
    time_log_interval: int = 50
    enable_tb: bool = False  # Enable TensorBoard logging

    # Performance optimization settings
    use_amp: bool = True  # Automatic Mixed Precision
    torch_compile: bool = False  # torch.compile optimization
    prefetch_factor: int = 4  # DataLoader prefetch factor
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    on_the_fly_sampling: bool = (
        True  # Use on-the-fly sampling instead of batch collection
    )

    # Checkpoint / output settings
    output_base_dir: str = "outputs"
    save_interval: int = 10


@attr.s(auto_attribs=True, frozen=True)
class RLConfig:
    """Configuration for reinforcement learning fine-tuning."""

    bc_config_path: str
    dataset: DatasetConfig
    network: NetworkConfig = NetworkConfig()
    ppo: PPOConfig = PPOConfig()
    sac: SACConfig = SACConfig()
    training: TrainingConfig = TrainingConfig()
    rollout_buffer: RolloutBufferConfig = RolloutBufferConfig()

    # Structured data processing settings
    enable_structured_data: bool = (
        True  # Whether to use structured state/action processing
    )
    ignore_depth_images: bool = True  # Whether to ignore depth images in processing
