# Bimanual A1 Galaxea Motion Planning Guide

This guide provides instructions for running motion planning with the A1 Galaxea robot in both single-arm and bimanual configurations using ManiSkill.

## Overview

The A1 Galaxea motion planning system has been enhanced to support bimanual operation while maintaining backward compatibility with single-arm tasks. The system automatically detects and configures the appropriate mode based on the environment.

### Supported Configurations

- **Single-Arm Mode**: Traditional single A1 Galaxea arm (PickCube-v1)
- **Bimanual Mode**: Dual A1 Galaxea arms with left arm static, right arm active (PickBox-v1)

## Prerequisites

1. **Environment Setup**:

   ```bash
   conda activate maniskill
   ```

2. **Navigate to Motion Planning Directory**:

   ```bash
   cd mani_skill/examples/motionplanning/a1_galaxea
   ```

## Quick Start

### Basic Commands

#### PickBox (Bimanual Mode)

```bash
# Basic bimanual pick-and-place with GUI
python run.py -e PickBox-v1 --vis

# Multiple trajectories with ray tracing
python run.py -e PickBox-v1 --vis --shader rt-fast -n 5
```

#### PickCube (Single-Arm Mode)

```bash
# Basic single-arm pick-and-place with GUI
python run.py -e PickCube-v1 --vis

# With video recording and ray tracing
python run.py -e PickCube-v1 --vis --save-video --shader rt-fast -n 3
```

## Command Line Options

### Core Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `-e, --env-id` | Environment to run | `PickCube-v1` | `PickBox-v1` |
| `--vis` | Show GUI during execution | `False` | `--vis` |
| `-n, --num-traj` | Number of trajectories | `10` | `-n 5` |
| `--shader` | Rendering quality | `default` | `--shader rt-fast` |

### Rendering Options

| Argument | Description | Options |
|----------|-------------|---------|
| `--shader` | Shader pack | `default`, `rt`, `rt-fast` |
| `--render-mode` | Render mode | `rgb_array`, `human` |
| `--save-video` | Save trajectory videos | Flag |

### Advanced Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--obs-mode` | Observation mode | `none` |
| `--sim-backend` | Simulation backend | `auto` |
| `--record-dir` | Output directory | `demos` |
| `--traj-name` | Custom trajectory name | Timestamp |

## Environment Details

### PickBox (Bimanual Mode)

**Environment**: `PickBoxBimanual-v1`
**Robot Configuration**: Dual A1 Galaxea arms
**Behavior**:

- Left arm: Static at home position
- Right arm: Active, performs pick-and-place
- Task: Pick box from table, place in basket

**Motion Sequence**:

1. Move to pre-grasp pose (5cm above box)
2. Descend and grasp box
3. Lift box 15cm to avoid collisions
4. Move horizontally to above basket
5. Lower into basket and release

### PickCube (Single-Arm Mode)

**Environment**: `PickCube-v1`
**Robot Configuration**: Single A1 Galaxea arm
**Task**: Pick cube, move to goal position

**Motion Sequence**:

1. Move to pre-grasp pose (3cm above cube)
2. Descend and grasp cube
3. Transport to goal location

## Technical Implementation

### Automatic Mode Detection

The system automatically detects the appropriate configuration:

```python
# PickBox-v1 → PickBoxBimanual-v1 (bimanual mode)
# PickCube-v1 → PickCube-v1 (single-arm mode)
```

### Motion Planner Features

- **Collision Avoidance**: Disabled for A1 Galaxea due to mesh file issues
- **Gripper Control**: Precise prismatic gripper control
- **Path Planning**: RRTConnect and screw motion planning
- **Error Recovery**: Robust failure handling

### Bimanual Action Handling

In bimanual mode, actions are formatted as dictionaries:

```python
{
    "a1_galaxea-0": left_arm_action,   # Static at home position
    "a1_galaxea-1": right_arm_action   # Active motion planning
}
```

## Example Workflows

### Development and Testing

```bash
# Quick test with single trajectory
python run.py -e PickBox-v1 --vis -n 1

# Debug mode with detailed output
python run.py -e PickBox-v1 --vis -n 1 --shader default
```

### Demonstration and Recording

```bash
# High-quality demo with ray tracing
python run.py -e PickBox-v1 --vis --shader rt-fast -n 5

# Video recording (single-arm only due to dict action compatibility)
python run.py -e PickCube-v1 --vis --save-video --shader rt-fast
```

### Batch Generation

```bash
# Generate multiple trajectories
python run.py -e PickBox-v1 -n 20 --record-dir my_demos

# Custom naming
python run.py -e PickBox-v1 --vis --traj-name bimanual_demo -n 5
```

## Camera System (Bimanual Mode)

When using bimanual environments with camera observations:

### Available Cameras

- `eoat_left_top`: Left arm end-effector camera (RGB + Depth)
- `eoat_right_top`: Right arm end-effector camera (RGB + Depth)
- `static_top`: Static overhead camera (RGB + Depth)

### Camera Usage Example

```bash
# Test camera observations (use separate scripts for camera demos)
python ../../../run_bimanual_pick_box_cameras.py
```

## Troubleshooting

### Common Issues

1. **Mesh File Warnings**: A1 Galaxea has known mesh file issues. The system automatically disables collision checking as a workaround.

2. **Recording Issues**: Trajectory recording is disabled for bimanual mode due to dict action space compatibility. Video recording works for single-arm mode only.

3. **GUI Not Showing**: Ensure you have a display available and X11 forwarding if using SSH.

### Performance Tips

- Use `--shader default` for faster execution during development
- Use `--shader rt-fast` for high-quality demonstrations
- Reduce `-n` (number of trajectories) for quick testing

### Success Rate Interpretation

- Success rates may appear low (0.00) due to strict success criteria
- Monitor the motion execution visually with `--vis` flag
- Check episode lengths to verify motion completion

## File Structure

```
mani_skill/examples/motionplanning/a1_galaxea/
├── run.py                    # Main execution script (updated for bimanual)
├── motionplanner.py         # Motion planner (bimanual support)
├── solutions/
│   ├── pick_box.py         # PickBox solver (bimanual compatible)
│   └── pick_cube.py        # PickCube solver (bimanual compatible)
└── README.md               # This file
```

## Related Files

- `run_bimanual_pick_box.py`: Standalone bimanual demo with random actions
- `run_bimanual_pick_box_cameras.py`: Camera observation demo
- `run_bimanual_pick_box_motionplanning.py`: Alternative motion planning script

## Advanced Configuration

### Custom Robot Poses

The system uses predefined home poses for bimanual configuration. These can be modified in the robot configuration files.

### Motion Planning Parameters

- Joint velocity limits: Configurable in motion planner
- Gripper states: `OPEN = -0.03`, `CLOSED = 0.025`
- Planning algorithms: RRTConnect (default), screw motion

### Environment Extensions

To add new bimanual environments:

1. Create bimanual version of the environment
2. Update the environment mapping in `run.py`
3. Add solver compatibility checks
4. Test with motion planning system

## Support

For issues or questions:

1. Check console output for detailed error messages
2. Verify environment setup with `conda activate maniskill`
3. Test with single trajectory first: `-n 1`
4. Use `--vis` flag to observe motion execution

## Future Enhancements

- Full trajectory recording support for bimanual mode
- Additional bimanual environments (PickCube bimanual variant)
- Enhanced collision checking with fixed mesh files
- Multi-arm coordination algorithms
