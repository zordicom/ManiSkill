# Plan to Fix CPU/GPU Inconsistency in pd_ee_pose Controller

## Problem Statement

**Current Behavior:**

The `pd_ee_pose` controller behaves differently in GPU vs CPU mode when `use_target=False`:

| Mode | IK Path | Absolute→Delta Conversion | IK Solver | Failure Handling |
|------|---------|---------------------------|-----------|------------------|
| **CPU** | via `compute_target_pose()` | Explicit (in `compute_target_pose`) | SAPIEN analytical | Returns `None` (robot freezes) |
| **GPU** | Direct tensor to IK | **Implicit (inside GPU IK)** | Levenberg-Marquardt | Always returns approximation |

**Impact:**

- Policies trained on GPU fail when deployed on CPU
- Users must know internal implementation details to avoid breakage
- Violates principle of least surprise

---

## Root Cause

**File:** `mani_skill/agents/controllers/pd_ee_pose.py`

**Line 115:**

```python
ik_via_target_pose = self.config.use_target or not self.scene.gpu_sim_enabled
```

This creates two different code paths:

1. **CPU mode:** Always uses `ik_via_target_pose=True` → calls `compute_target_pose()`
2. **GPU mode with `use_target=False`:** Uses `ik_via_target_pose=False` → bypasses `compute_target_pose()`

**File:** `mani_skill/agents/controllers/utils/kinematics.py`

**Lines 207-221:** GPU IK automatically converts absolute → delta (duplicates `compute_target_pose` logic!)

---

## Proposed Solution

### Option 1: Always Use `compute_target_pose()` (Recommended)

**Change:** Remove the special GPU path optimization

**File:** `pd_ee_pose.py` line 115

**Before:**

```python
ik_via_target_pose = self.config.use_target or not self.scene.gpu_sim_enabled
```

**After:**

```python
# ALWAYS compute target pose explicitly for consistency between CPU/GPU
ik_via_target_pose = True
```

**Impact:**

- ✅ CPU and GPU behave identically
- ✅ No code duplication (absolute→delta logic only in `compute_target_pose`)
- ✅ Easier to understand and maintain
- ⚠️  Minor performance hit on GPU (one extra function call)
- ⚠️  May break existing code that relies on GPU-specific behavior

**Backward Compatibility:**

- Add deprecation warning if GPU mode is detected with `use_target=False`
- Provide migration guide for existing GPU-trained policies

---

### Option 2: Make CPU Use Same Path as GPU

**Change:** Make CPU bypass `compute_target_pose()` like GPU does

**File:** `pd_ee_pose.py` line 115

**Before:**

```python
ik_via_target_pose = self.config.use_target or not self.scene.gpu_sim_enabled
```

**After:**

```python
# Only use target pose path if explicitly requested
ik_via_target_pose = self.config.use_target
```

**File:** `kinematics.py` - Extract CPU IK absolute→delta logic to match GPU

**Impact:**

- ✅ CPU and GPU behave identically
- ✅ No performance regression
- ⚠️  Requires modifying CPU IK solver to handle tensor input
- ⚠️  More complex: CPU and GPU IK solvers need same interface

---

### Option 3: Add Explicit Warning/Error

**Change:** Detect the inconsistency and warn users

**File:** `pd_ee_pose.py` after line 137

**Add:**

```python
# After IK computation
if self._target_qpos is None and not self.scene.gpu_sim_enabled:
    logger.warning(
        f"IK FAILED in CPU mode. Note: Policies trained with GPU "
        f"(sim_backend='physx_cuda') may not work in CPU mode due to "
        f"different IK solvers. Consider using sim_backend='physx_cuda' "
        f"for deployment, or retrain with sim_backend='cpu'."
    )
```

**Impact:**

- ✅ Easy to implement
- ✅ No behavioral change
- ✅ Users are informed of the issue
- ❌ Doesn't actually fix the problem

---

### Option 4: Add `use_delta` Override (Best for Long-Term)

**Change:** Make `use_delta` the recommended mode, deprecate absolute pose mode

**Files:**

- `pd_ee_pose.py`: Make `use_delta=True` the default
- Documentation: Recommend delta mode for portability
- Baselines: Update PPO/SAC examples to use `use_delta=True`

**Impact:**

- ✅ CPU and GPU behave identically with `use_delta=True`
- ✅ More intuitive (deltas are easier to learn)
- ✅ Better policy portability
- ⚠️  Breaking change (requires version bump)
- ⚠️  Existing checkpoints won't work

---

## Recommendation

**Short-term (v3.x):** Implement **Option 1 + Option 3**

1. Change line 115 to always use `compute_target_pose()` (unify behavior)
2. Add deprecation warning for GPU-specific path
3. Update documentation to explain the change

**Long-term (v4.0):** Implement **Option 4**

1. Make `use_delta=True` the default
2. Deprecate absolute pose mode entirely
3. Update all baselines and examples

---

## Implementation Steps

### Phase 1: Quick Fix (Option 1)

1. **File:** `mani_skill/agents/controllers/pd_ee_pose.py`

   ```python
   # Line 115 - BEFORE:
   ik_via_target_pose = self.config.use_target or not self.scene.gpu_sim_enabled

   # Line 115 - AFTER:
   # UNIFIED BEHAVIOR: Always use compute_target_pose for CPU/GPU consistency
   # This fixes the issue where policies trained on GPU fail on CPU
   ik_via_target_pose = True

   # Add warning for users relying on old GPU behavior
   if not self.config.use_target and self.scene.gpu_sim_enabled:
       logger.warning(
           f"{self.__class__.__name__}: GPU-specific IK path is deprecated. "
           "Future versions will unify CPU/GPU behavior. "
           "To suppress this warning, set use_target=True explicitly."
       )
   ```

2. **File:** `docs/source/user_guide/concepts/controllers.rst` (or equivalent)

   Add section explaining:
   - The CPU/GPU difference (before fix)
   - Why it matters for policy deployment
   - Migration guide for existing GPU-trained policies

3. **File:** `examples/baselines/ppo/README.md` (or similar)

   Add note:

   ```markdown
   ## Important: sim_backend Consistency

   When deploying trained policies, **always use the same sim_backend as training**:
   - Trained with `sim_backend="physx_cuda"` → Deploy with `"physx_cuda"`
   - Trained with `sim_backend="cpu"` → Deploy with `"cpu"`

   Starting from ManiSkill v3.x, this requirement is removed (CPU/GPU unified).
   ```

4. **Testing:**
   - Test that existing GPU-trained policies still work
   - Test that CPU mode now works with same policies
   - Verify no performance regression

### Phase 2: Long-term Fix (Option 4)

1. Change default: `use_delta: bool = True` (in v4.0)
2. Add migration script to convert old checkpoints
3. Update all baselines/examples

---

## Testing Plan

Create test script to verify CPU/GPU consistency:

```python
# tests/test_cpu_gpu_consistency.py

import gymnasium as gym
import torch
import numpy as np

def test_cpu_gpu_pd_ee_pose_consistency():
    """Test that pd_ee_pose behaves identically in CPU and GPU mode."""

    action = np.array([0.35, -0.69, -0.52, -1.22, -0.91, 0.07, 0.79], dtype=np.float32)

    # CPU mode
    env_cpu = gym.make("PickCube-v1", control_mode="pd_ee_pose", obs_mode="state", sim_backend="cpu")
    obs_cpu, _ = env_cpu.reset(seed=42)
    obs_cpu_after, _, _, _, _ = env_cpu.step(action)
    qpos_cpu_after = obs_cpu_after[0, :7]
    env_cpu.close()

    # GPU mode
    env_gpu = gym.make("PickCube-v1", control_mode="pd_ee_pose", obs_mode="state", sim_backend="physx_cuda")
    obs_gpu, _ = env_gpu.reset(seed=42)
    obs_gpu_after, _, _, _, _ = env_gpu.step(action)
    qpos_gpu_after = obs_gpu_after[0, :7]
    env_gpu.close()

    # Compare
    assert torch.allclose(qpos_cpu_after, qpos_gpu_after, atol=1e-3), \
        f"CPU and GPU produced different results!\nCPU: {qpos_cpu_after}\nGPU: {qpos_gpu_after}"

    print("✅ CPU and GPU pd_ee_pose behave consistently!")
```

---

## Files to Modify

1. `mani_skill/agents/controllers/pd_ee_pose.py` (line 115)
2. `mani_skill/agents/controllers/utils/kinematics.py` (add warning/docs)
3. `docs/source/user_guide/concepts/controllers.rst`
4. `examples/baselines/ppo/README.md`
5. `tests/test_cpu_gpu_consistency.py` (new file)
6. `CHANGELOG.md` (document the fix)

---

## Migration Guide for Users

### If You Have GPU-Trained Policies

**Before (broken):**

```python
# Training
env = gym.make("PickCube-v1", sim_backend="physx_cuda", control_mode="pd_ee_pose")
train_ppo(env)  # Trains successfully

# Deployment
env = gym.make("PickCube-v1", sim_backend="cpu", control_mode="pd_ee_pose")
run_policy(env, checkpoint)  # ❌ FAILS - robot doesn't move!
```

**After (fixed):**

```python
# Deployment (post-fix)
env = gym.make("PickCube-v1", sim_backend="cpu", control_mode="pd_ee_pose")
run_policy(env, checkpoint)  # ✅ WORKS - CPU now behaves like GPU!
```

### For Future Training

**Recommended:**

```python
# Use delta mode for portability
controller_config = PDEEPoseControllerConfig(
    use_delta=True,  # Portable between CPU/GPU!
    ...
)
```

---

## Priority

**HIGH** - This affects policy deployment and is a common user confusion point.

The fix is simple (1 line change + tests) but has broad impact on ecosystem.

---

## Questions for ManiSkill Maintainers

1. Was the GPU-specific path intentional or accidental?
2. Are there performance benchmarks showing the GPU optimization is necessary?
3. Should we add a config flag to opt-in to old behavior for backward compat?
4. Timeline for v4.0 with `use_delta=True` as default?

---

## References

- Issue discussion: [Link to GitHub issue if created]
- Related: <https://github.com/haosulab/ManiSkill/issues/955> (IK solver performance)
- Code: `agents/controllers/pd_ee_pose.py` lines 107-137
- Code: `agents/controllers/utils/kinematics.py` lines 204-260
