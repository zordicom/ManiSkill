# ManiSkill Environment Wrappers

This directory contains environment wrappers for ManiSkill that modify or extend the functionality of base environments.

## Available Wrappers

### ExpertResidualWrapper

The `ExpertResidualWrapper` enables hybrid control by combining expert policies with learned residual corrections. This is useful for leveraging domain knowledge while still allowing learning agents to improve performance.

**Key Features:**
- Combines expert baseline actions with learned residual corrections
- Extends observation space to include expert actions
- Supports any expert policy (IK solvers, heuristics, pre-trained models)
- Configurable residual scaling and action clipping
- Full torch tensor support for ManiSkill3

**Usage:**
```python
from mani_skill.envs.wrappers import ExpertResidualWrapper

# Define expert policy
def ik_expert(obs):
    # Your expert policy logic here
    return expert_action

# Wrap environment
wrapped_env = ExpertResidualWrapper(
    env,
    expert_policy_fn=ik_expert,
    residual_scale=0.1,
    expert_action_in_obs=True,
)

# Train residual agent
obs, info = wrapped_env.reset()
residual_action = agent.select_action(obs)
obs, reward, terminated, truncated, info = wrapped_env.step(residual_action)
```

See `docs/expert_residual_wrapper.md` for detailed documentation.

### SuccessRateCurriculumWrapper

The `SuccessRateCurriculumWrapper` provides automatic curriculum learning based on success rates. It progressively increases task difficulty as the agent improves.

**Key Features:**
- Automatic curriculum progression based on success rates
- Configurable success thresholds and episode requirements
- Per-level time limits and episode step management
- Comprehensive curriculum statistics tracking

**Usage:**
```python
from mani_skill.envs.wrappers import SuccessRateCurriculumWrapper

curriculum_levels = [
    {"name": "Easy", "max_episode_steps": 50, "difficulty": 0.5},
    {"name": "Medium", "max_episode_steps": 75, "difficulty": 0.7},
    {"name": "Hard", "max_episode_steps": 100, "difficulty": 1.0},
]

wrapped_env = SuccessRateCurriculumWrapper(
    env,
    curriculum_levels=curriculum_levels,
    success_threshold=0.8,
    window_size=100,
)
```

## Creating Custom Wrappers

When creating custom wrappers for ManiSkill:

1. **Inherit from `gym.Wrapper`**: ManiSkill environments are compatible with Gymnasium
2. **Use torch tensors**: All observations and actions should be torch tensors
3. **Handle batched environments**: Support `num_envs > 1` for parallel training
4. **Preserve device**: Keep tensors on the same device as the environment
5. **Update spaces**: Properly modify observation/action spaces if needed

**Example:**
```python
import gymnasium as gym
import torch
from mani_skill.envs.sapien_env import BaseEnv

class CustomWrapper(gym.Wrapper):
    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.device = env.device
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Modify observation here
        return obs, info
        
    def step(self, action):
        # Ensure action is torch tensor on correct device
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Modify outputs here
        return obs, reward, terminated, truncated, info
```

## Testing

Run tests for the wrappers:
```bash
python -m pytest tests/test_expert_residual_wrapper.py -v
```

## Examples

See the `examples/` directory for complete working examples:
- `expert_residual_example.py`: Expert+Residual wrapper usage
- Additional examples coming soon

## Contributing

When adding new wrappers:
1. Add the wrapper to `__init__.py`
2. Include comprehensive docstrings
3. Add unit tests
4. Update this README
5. Add example usage if applicable 