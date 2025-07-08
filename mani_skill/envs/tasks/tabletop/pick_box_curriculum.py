"""
Copyright 2025 Zordi, Inc. All rights reserved.

Curriculum configuration for PickBox task.
"""

# Curriculum levels for PickBox task
PICK_BOX_CURRICULUM_LEVELS = [
    {
        # Level 1: Basic - just a cube, 50-timesteps per episode
        "name": "Basic Cube",
        "description": "Simple uniform cube with 50 timesteps",
        "max_episode_steps": 50,
        "box_dimensions": {
            "type": "cube",
            "size": 0.02,  # 2cm half-size
        },
        "use_real_b5box_probability": 0.0,  # Only primitive boxes
        "physics_variation": False,
        "visual_variation": False,
    },
    {
        # Level 2: Non-uniform cube (x-axis 3x longer), 50-timestep
        "name": "Non-uniform Cube",
        "description": "Rectangular box with x-axis 3x longer, 50 timesteps",
        "max_episode_steps": 50,
        "box_dimensions": {
            "type": "non_uniform",
            "base_size": 0.02,
            "x_multiplier": 3.0,
        },
        "use_real_b5box_probability": 0.0,  # Only primitive boxes
        "physics_variation": False,
        "visual_variation": False,
    },
    {
        # Level 3: Randomized dimensions, 50 timestep
        "name": "Randomized Dimensions",
        "description": "Randomized box dimensions, 50 timesteps",
        "max_episode_steps": 50,
        "box_dimensions": {
            "type": "randomized",
            "base_size": 0.02,
        },
        "use_real_b5box_probability": 0.0,  # Only primitive boxes
        "physics_variation": False,
        "visual_variation": False,
    },
    {
        # Level 4: 30% probability of b5box, 50 timestep
        "name": "Mixed Assets",
        "description": "30% real b5box assets, 70% primitive boxes, 50 timesteps",
        "max_episode_steps": 50,
        "box_dimensions": {
            "type": "randomized",
            "base_size": 0.02,
        },
        "use_real_b5box_probability": 0.3,  # 30% real b5box
        "physics_variation": False,
        "visual_variation": False,
    },
    {
        # Level 5: Add other physics variation, 50 timesteps
        "name": "Physics Variation",
        "description": "Physics variations (friction, mass, damping), 50 timesteps",
        "max_episode_steps": 50,
        "box_dimensions": {
            "type": "randomized",
            "base_size": 0.02,
        },
        "use_real_b5box_probability": 0.3,
        "physics_variation": True,
        "visual_variation": False,
    },
    {
        # Level 6: Now add more timesteps, 150 timesteps (per-episode max length)
        "name": "Extended Episodes",
        "description": "Longer episodes with 150 timesteps, physics variations",
        "max_episode_steps": 150,
        "box_dimensions": {
            "type": "randomized",
            "base_size": 0.02,
        },
        "use_real_b5box_probability": 0.3,
        "physics_variation": True,
        "visual_variation": False,
    },
    {
        # Level 7: Add visual variation
        "name": "Full Complexity",
        "description": "All variations: physics, visual, 150 timesteps",
        "max_episode_steps": 150,
        "box_dimensions": {
            "type": "randomized",
            "base_size": 0.02,
        },
        "use_real_b5box_probability": 0.3,
        "physics_variation": True,
        "visual_variation": True,
    },
]


def get_curriculum_levels():
    """Get the curriculum levels for PickBox task."""
    return PICK_BOX_CURRICULUM_LEVELS.copy()


def create_curriculum_wrapper(env, **kwargs):
    """Create a curriculum wrapper for PickBox task.

    Args:
        env: The PickBox environment to wrap
        **kwargs: Additional arguments for SuccessRateCurriculumWrapper

    Returns:
        SuccessRateCurriculumWrapper: The wrapped environment
    """
    from mani_skill.envs.wrappers import SuccessRateCurriculumWrapper

    # Default curriculum wrapper settings
    default_kwargs = {
        "success_threshold": 0.8,
        "window_size": 100,
        "min_episodes_per_level": 1000,
        "steps_per_level": 1_000_000,
        "verbose": True,
    }

    # Override with user-provided kwargs
    default_kwargs.update(kwargs)

    return SuccessRateCurriculumWrapper(
        env=env, curriculum_levels=get_curriculum_levels(), **default_kwargs
    )
