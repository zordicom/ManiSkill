# Pick Box Dataset Generator

A comprehensive dataset generator for training manipulation policies using the A1 Galaxea robot performing pick-and-place tasks with boxes.

## Overview

This generator creates high-quality training datasets by recording A1 Galaxea robot trajectories during pick-and-place operations. It uses motion planning to generate expert demonstrations with multi-modal sensor data.

## Features

- **Multi-Camera Recording**: RGB, depth, and segmentation from multiple viewpoints
- **Bimanual Support**: Works with both single-arm and bimanual A1 Galaxea configurations
- **Coordinate System Options**: World coordinates or table-origin relative coordinates
- **Video Generation**: Automatic MP4 compilation for all camera views
- **Rich Metadata**: Camera intrinsics, segmentation mappings, and episode information

## Generated Data Structure

```
~/training_data/galaxea_box_pnp_sim/
└── galaxea_box_pnp/
    └── galaxea-16hz_box_pnp/
        └── data/
            └── YYYY/MM/DD/
                └── episode_timestamp/
                    ├── episode_timestamp.json       # Episode metadata & observations
                    ├── static_top_rgb/             # Overhead camera RGB
                    ├── static_top_depth/           # Overhead camera depth
                    ├── static_top_segmentation/    # Overhead camera segmentation
                    ├── eoat_left_top_rgb/          # Left gripper camera RGB
                    ├── eoat_right_top_rgb/         # Right gripper camera RGB
                    ├── eoat_*_depth/               # Gripper cameras depth
                    ├── eoat_*_segmentation/        # Gripper cameras segmentation
                    └── *.mp4                       # Compiled videos
```

## Quick Start

### Basic Usage
```bash
# Generate 3 episodes with default settings
python pick_box_dataset_generator.py

# Generate 10 episodes with specific output directory
python pick_box_dataset_generator.py \
    --n-episodes 10 \
    --output-dir ~/my_datasets/pick_box/
```

### Environment Options
```bash
# Single-arm mode
python pick_box_dataset_generator.py \
    --env-id PickBox-v1 \
    --robot-uids a1_galaxea

# Bimanual mode  
python pick_box_dataset_generator.py \
    --env-id PickBoxBimanual-v1 \
    --robot-uids a1_galaxea
```

### Data Format Options
```bash
# Use world coordinates
python pick_box_dataset_generator.py \
    --no-table-origin

# Use table-origin coordinates (default)
python pick_box_dataset_generator.py \
    --use-table-origin
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--n-episodes` | Number of episodes to generate | `3` |
| `--output-dir` | Root directory for dataset | `~/training_data/galaxea_box_pnp_sim/` |
| `--seed-start` | Starting seed for reproducibility | `42` |
| `--env-id` | Environment ID | `PickBoxBimanual-v1` |
| `--robot-uids` | Robot configuration | `a1_galaxea` |
| `--obs-mode` | Observation mode | `rgb+depth+segmentation` |
| `--control-mode` | Control mode | `pd_joint_pos` |
| `--use-table-origin` | Use table-origin coordinates | `True` |
| `--debug` | Enable debug output | `False` |
| `--vis` | Enable visualization | `False` |
| `--no-video` | Disable video recording | `False` |
| `--video-fps` | Video frame rate | `30` |

## Data Format

### Episode JSON Structure
```json
{
  "metadata": {
    "episode_name": "20250101_120000",
    "robot_id": "a1_galaxea",
    "k_mats": {
      "static_top": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
      "eoat_left_top": [...],
      "eoat_right_top": [...]
    },
    "camera_image_dimensions": {
      "static_top": {"width": 640, "height": 480}
    },
    "coordinate_system": {
      "type": "table_origin",
      "table_origin": [x, y, z],
      "right_arm_offset": [-0.025, -0.365, 0.005],
      "left_arm_offset": [-0.025, 0.365, 0.005]
    },
    "segmentation_legend": {
      "segmentation_mapping": {...},
      "color_mapping": {...},
      "id_remapping": {...}
    }
  },
  "observations": [
    {
      "frame_index": 0,
      "delta_ms": 0.0,
      "elapsed_ms": 0.0,
      "joint_states_left": [7 joint values],
      "joint_states_right": [7 joint values],
      "tool_pose_left": [x, y, z, qx, qy, qz, qw],
      "tool_pose_right": [x, y, z, qx, qy, qz, qw],
      "pick_target_pose": [x, y, z, qx, qy, qz, qw],
      "place_target_pose": [x, y, z, qx, qy, qz, qw],
      "static_top_rgb": "static_top_rgb/000000.jpg",
      "static_top_depth": "static_top_depth/000000.jpg",
      "static_top_segmentation": "static_top_segmentation/000000.jpg",
      "eoat_left_top_rgb": "eoat_left_top_rgb/000000.jpg",
      "eoat_right_top_rgb": "eoat_right_top_rgb/000000.jpg"
    }
  ]
}
```

### Coordinate Systems

#### Table-Origin Coordinates (Default)
- Origin at table center between robot arms
- Right arm at `[-0.025, -0.365, 0.005]` relative to table origin
- Left arm at `[-0.025, 0.365, 0.005]` relative to table origin
- Better for learning arm-relative policies

#### World Coordinates
- Standard ManiSkill world coordinate system
- Direct from environment without transformation
- Better for absolute positioning tasks

### Segmentation Mapping
- **ID 0**: Background
- **ID 1**: Object to pick (box/cube)
- **ID 2**: Target container (basket)
- **ID 3**: Robot gripper/fingers
- **ID 4**: Table/surface
- **ID 5**: Goal markers (if any)

## Examples

### Generate Development Dataset
```bash
python pick_box_dataset_generator.py \
    --n-episodes 5 \
    --debug \
    --vis \
    --output-dir ~/dev_data/
```

### Generate Production Dataset
```bash
python pick_box_dataset_generator.py \
    --n-episodes 100 \
    --output-dir ~/production_data/ \
    --seed-start 1000 \
    --no-video  # Skip video for faster generation
```

### Generate Bimanual Dataset
```bash
python pick_box_dataset_generator.py \
    --env-id PickBoxBimanual-v1 \
    --n-episodes 50 \
    --output-dir ~/bimanual_data/
```

## Use Cases

- **Imitation Learning**: Train policies from expert demonstrations
- **Behavior Cloning**: Learn pick-and-place behaviors
- **Data Augmentation**: Generate diverse training scenarios
- **Policy Evaluation**: Benchmark learned policies against expert performance
- **Multi-Modal Learning**: Train with RGB, depth, and segmentation data

## Requirements

- ManiSkill environment
- A1 Galaxea robot configuration
- OpenCV for image processing
- Motion planning dependencies

## Troubleshooting

### Common Issues

**Motion Planning Fails**
- Check robot configuration and environment setup
- Verify grasping pose computation
- Enable `--debug` for detailed output

**Missing Camera Data**
- Ensure observation mode includes required sensors
- Check camera configurations in environment

**File Permission Errors**
- Verify write permissions for output directory
- Use absolute paths for reliability

Happy dataset generation! 🤖📊 