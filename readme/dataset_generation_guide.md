# A1 Galaxea Dataset Generation Guide

> **Comprehensive guide for generating training datasets using the A1 Galaxea robot with multi-modal observations (RGB, depth, segmentation)**

---

## Overview

The A1 Galaxea dataset generator provides a complete solution for creating training datasets with:

- **Frame-by-frame capture** at every control step (not just keyframes)
- **Dual camera perspectives** (static overhead + gripper-mounted)
- **Multi-modal data** (RGB, depth, colorized segmentation, raw segmentation)
- **Video recording** from default human render camera (compiled as MP4)
- **Intelligent segmentation** with predetermined colors and simplified IDs
- **Comprehensive metadata** for each episode

---

## Quick Start

### 1. Environment Setup

```bash
# Activate ManiSkill environment
conda activate maniskill

# Navigate to ManiSkill directory
cd /path/to/ManiSkill
```

### 2. Generate Dataset with Default Settings

```bash
# Generate 3 episodes with default settings
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py
```

This will create a dataset in `./training_data/` with 3 episodes using seeds 42, 43, 44.

### 3. Generate Custom Number of Episodes

```bash
# Generate 10 episodes
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py --n-episodes 10

# Generate 50 episodes with custom output directory
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py --n-episodes 50 --output-dir /path/to/dataset

# Generate with visualization and debug output
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py --n-episodes 5 --vis --debug

# Generate without video recording for faster processing
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py --n-episodes 100 --no-video
```

---

## Command Line Interface

### Available Options

```bash
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py --help
```

**Key CLI Arguments:**

- `--n-episodes`: Number of episodes to generate (default: 3)
- `--output-dir`: Root directory for dataset output (default: ./training_data)
- `--seed-start`: Starting seed for episodes (default: 42)
- `--debug`: Enable debug output
- `--vis`: Enable visualization (opens GUI)
- `--no-video`: Disable video recording (faster processing)
- `--video-fps`: Frame rate for compiled video (default: 30)
- `--env-id`: Environment ID (PickBox-v1 or PickBoxBimanual-v1, default: PickBoxBimanual-v1)
- `--robot-uids`: Robot UIDs to use (default: a1_galaxea)
- `--obs-mode`: Observation mode (default: rgb+depth+segmentation)
- `--control-mode`: Control mode (default: pd_joint_pos)

### Example Commands

```bash
# Generate large dataset efficiently
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py \
    --n-episodes 1000 \
    --output-dir /data/a1_dataset \
    --seed-start 1000 \
    --no-video

# Generate with single-arm mode
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py \
    --n-episodes 10 \
    --env-id PickBox-v1

# Generate with visualization for debugging
python mani_skill/examples/motionplanning/a1_galaxea/solutions/pick_box_dataset_generator.py \
    --n-episodes 5 \
    --vis \
    --debug \
    --video-fps 60
```

---

## Custom Dataset Generation

### Basic Python Usage

```python
import gymnasium as gym
import mani_skill
from mani_skill.examples.motionplanning.a1_galaxea.solutions.pick_box_dataset_generator import generate_pick_box_dataset

# Create environment with full multimodal observations
env = gym.make(
    'PickBox-v1',
    robot_uids='a1_galaxea',
    obs_mode='rgb+depth+segmentation',  # Essential for full dataset
    render_mode='rgb_array',
    control_mode='pd_joint_pos',
)

# Generate dataset
generate_pick_box_dataset(
    env=env,
    num_episodes=50,                    # Number of episodes
    output_root_dir='./training_data',  # Output directory
    seed_start=100,                     # Starting seed
    debug=False,                        # Debug output
    vis=False,                          # Visualization
    save_video=True,                    # Video recording (default: True)
    video_fps=30                        # Video frame rate (default: 30)
)

env.close()
```

### Advanced Configuration

```python
# Large dataset generation
generate_pick_box_dataset(
    env=env,
    num_episodes=1000,                  # Large dataset
    output_root_dir='/data/a1_dataset', # Custom path
    seed_start=1000,                    # Different seed range
    debug=True,                         # Enable debug output
    vis=False,                          # Keep visualization off for speed
    save_video=True,                    # Video recording enabled
    video_fps=30                        # Standard video frame rate
)
```

### Visualization Mode

```python
# Generate with visualization (slower but useful for debugging)
generate_pick_box_dataset(
    env=env,
    num_episodes=5,
    output_root_dir='./debug_data',
    seed_start=0,
    debug=True,
    vis=True,  # Enable GUI visualization
    save_video=True,  # Video recording enabled
    video_fps=30      # Standard video frame rate
)
```

---

## Dataset Structure

### Directory Layout

```
training_data/
└── galaxea_box_pnp/
    └── galaxea-16hz_box_pnp/
        └── data/
            └── 2025/07/08/
                └── 20250708_143052/                    # Episode directory
                    ├── 20250708_143052.json            # Frame-by-frame data
                    ├── 20250708_143052_default_camera.mp4 # Default camera video
                    ├── 20250708_143052_static_top_camera.mp4 # Static top camera video
                    ├── 20250708_143052_eoat_camera.mp4 # End effector camera video
                    ├── metadata.json                   # Segmentation mappings
                    ├── static_top_rgb/                 # RGB overhead camera
                    │   ├── 000000.jpg
                    │   ├── 000001.jpg
                    │   └── ...
                    ├── static_top_depth/               # Depth overhead camera
                    │   ├── 000000.jpg
                    │   └── ...
                    ├── static_top_segmentation/        # Colorized segmentation
                    │   ├── 000000.jpg
                    │   └── ...
                    ├── static_top_segmentation_raw/    # Raw PNG (IDs 0-5)
                    │   ├── 000000.png
                    │   └── ...
                    ├── default_camera_video/           # Default camera frames
                    │   ├── 000000.jpg
                    │   ├── 000001.jpg
                    │   └── ...
                    ├── eoat_right_top_rgb/            # RGB gripper camera
                    ├── eoat_right_top_depth/          # Depth gripper camera
                    ├── eoat_right_top_segmentation/   # Colorized segmentation
                    ├── eoat_right_top_segmentation_raw/ # Raw PNG (IDs 0-5)
                    └── eoat_left_top_rgb/             # Duplicate for compatibility
```

### Episode Data Format

**Frame Data (`episode_name.json`):**

```json
[
  {
    "episode_name": "20250708_143052",
    "frame_index": 0,
    "delta_ms": 62.5,
    "elapsed_ms": 0.0,
    "state": [0.733, 0.785, -0.838, 1.396, -0.611, 0.262, -0.03, -0.03],
    "action": [0.733, 0.785, -0.838, 1.396, -0.611, 0.262, -0.03, -0.03],
    "task": "pick and place the box",
    "static_top_rgb": "static_top_rgb/000000.jpg",
    "static_top_depth": "static_top_depth/000000.jpg",
    "static_top_segmentation": "static_top_segmentation/000000.jpg",
    "static_top_segmentation_raw": "static_top_segmentation_raw/000000.png",
    "eoat_right_top_rgb": "eoat_right_top_rgb/000000.jpg",
    "phase": "initial"
  }
]
```

**Metadata (`metadata.json`):**

```json
{
  "episode_name": "20250708_143052",
  "total_frames": 184,
  "segmentation_mapping": {
    "14": {"type": "actor", "name": "cube_0", "description": "Actor: cube_0"},
    "15": {"type": "actor", "name": "basket", "description": "Actor: basket"},
    "9": {"type": "link", "name": "left_finger", "description": "Link: left_finger"}
  },
  "id_remapping": {
    "14": 1, "15": 2, "9": 3, "10": 3, "13": 4, "16": 5
  },
  "simplified_id_legend": {
    "0": "Background/other objects",
    "1": "Object to pick (cube/box/b5box)",
    "2": "Target container (basket)",
    "3": "Robot gripper/fingers",
    "4": "Surface (table/ground)",
    "5": "Goal/target markers"
  }
}
```

---

## Segmentation System

### Simplified ID Mapping

The dataset uses a simplified segmentation system perfect for training:

| ID | Object Type | Color (BGR) | Description |
|----|-------------|-------------|-------------|
| **0** | Background | Black [0,0,0] | Background/other objects |
| **1** | Target Object | Red [0,0,255] | Object to pick (cube/box) |
| **2** | Container | Green [0,255,0] | Target container (basket) |
| **3** | Gripper | Blue [255,0,0] | Robot gripper/fingers |
| **4** | Surface | Gray [128,128,128] | Table/ground surface |
| **5** | Goal Marker | Variable | Goal/target markers |

### Two Segmentation Formats

1. **Colorized Segmentation (JPG)**: For visualization and debugging
   - Uses predetermined colors for easy identification
   - Saved as `*_segmentation/*.jpg`

2. **Raw Segmentation (PNG)**: For training
   - UINT8 values with IDs 0-5
   - Saved as `*_segmentation_raw/*.png`
   - Perfect for semantic segmentation models

---

## Camera Setup

### Dual Camera System

**Static Top Camera (`static_top_*`):**

- **Position**: Overhead view of workspace
- **Purpose**: Scene context and global manipulation view
- **Resolution**: 128x128 (configurable)
- **Modalities**: RGB, depth, segmentation

**End Effector Camera (`eoat_right_top_*`):**

- **Position**: Gripper-mounted camera
- **Purpose**: Detailed manipulation view
- **Resolution**: 128x128 (configurable)
- **Modalities**: RGB, depth, segmentation

### Camera Data Access

```python
# Example: Loading camera data from episode
import cv2
import numpy as np

# Load RGB image
rgb_image = cv2.imread('static_top_rgb/000050.jpg')

# Load raw segmentation
seg_raw = cv2.imread('static_top_segmentation_raw/000050.png', cv2.IMREAD_GRAYSCALE)

# seg_raw now contains values 0-5 representing different object classes

# Load video frame from default camera
video_frame = cv2.imread('default_camera_video/000050.jpg')
```

### Video Recording

**Multi-Camera Video Recording (RGB + Depth + Segmentation):**

The generator now produces *nine* sensor-specific videos **in addition** to the default overview camera:

| MP4 file | Source camera | Modality |
|----------|---------------|----------|
| `{episode}_static_top_camera.mp4` | `static_top` | RGB |
| `{episode}_static_top_depth.mp4` | `static_top` | Depth (colorized) |
| `{episode}_static_top_segmentation.mp4` | `static_top` | Segmentation (colorized) |
| `{episode}_eoat_left_top_camera.mp4` | `eoat_left_top` | RGB |
| `{episode}_eoat_left_top_depth.mp4` | `eoat_left_top` | Depth (colorized) |
| `{episode}_eoat_left_top_segmentation.mp4` | `eoat_left_top` | Segmentation (colorized) |
| `{episode}_eoat_right_top_camera.mp4` | `eoat_right_top` | RGB |
| `{episode}_eoat_right_top_depth.mp4` | `eoat_right_top` | Depth (colorized) |
| `{episode}_eoat_right_top_segmentation.mp4` | `eoat_right_top` | Segmentation (colorized) |
| `{episode}_default_camera.mp4` | Default GUI camera | RGB |

All videos are compiled at **30 FPS** (configurable via `video_fps`).

> **Tip:** If you only need RGB videos you can disable depth / segmentation in code by turning off the relevant video-buffer appends.

```python
# Quick preview helper
import cv2, glob
for mp4 in glob.glob('episode_dir/*_camera.mp4'):
    print('Playing', mp4)
    cap = cv2.VideoCapture(mp4)
    while cap.isOpened():
        ok, frame = cap.read();
        if not ok: break
        cv2.imshow(mp4, frame)
        if cv2.waitKey(1)&0xFF==27: break  # Esc to quit
    cap.release()
    cv2.destroyAllWindows()
```

---

## Live-render Demo (no dataset saving)

To execute the same motion-planning logic interactively without saving frames:

```bash
conda activate maniskill
python - << 'PY'
import gymnasium as gym, mani_skill
from mani_skill.examples.motionplanning.a1_galaxea.motionplanner import A1GalaxeaMotionPlanningSolver

# Bimanual environment with GUI
env = gym.make(
    'PickBoxBimanual-v1',
    robot_uids='a1_galaxea',
    obs_mode='state',          # no heavy images
    render_mode='human',       # opens GUI
    control_mode='pd_joint_pos'
)
env.reset()

# Create planner (vis=True draws target poses)
planner = A1GalaxeaMotionPlanningSolver(env, vis=True, debug=True)

# Example: open gripper then close after 1 s
planner.open_gripper(); planner.close_gripper()

input('Press Enter to quit…')
env.close()
PY
```

- Set `render_mode='human'` to open a realtime viewer.
- Keep `obs_mode='state'` for maximum FPS when you do not need vision data.
- All existing solver methods (`move_to_pose_with_RRTConnect`, `follow_path`, …) work unchanged.

---

## Performance & Optimization

### Generation Speed

- **Typical episode**: ~180-200 frames
- **Generation time**: ~2-3 minutes per episode
- **Disk usage**: ~50-100MB per episode (depends on image compression)
- **Video files**: ~15-45MB per episode (3 MP4 files: default, static_top, eoat cameras)

### Optimization Tips

```python
# For faster generation (disable visualization)
generate_pick_box_dataset(
    env=env,
    num_episodes=100,
    vis=False,      # Disable GUI
    debug=False,    # Disable debug output
    save_video=True, # Video recording still enabled
    video_fps=30    # Standard frame rate
)

# For debugging (enable visualization)
generate_pick_box_dataset(
    env=env,
    num_episodes=5,
    vis=True,       # Enable GUI
    debug=True,     # Enable debug output
    save_video=True, # Video recording enabled
    video_fps=30    # Standard frame rate
)
```

---

## Training Integration

### Loading Data for Training

```python
import json
import cv2
from pathlib import Path

def load_episode_data(episode_dir):
    """Load complete episode data."""
    episode_dir = Path(episode_dir)

    # Load frame data
    with open(episode_dir / f"{episode_dir.name}.json") as f:
        frame_data = json.load(f)

    # Load metadata
    with open(episode_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load images for specific frame
    frame_idx = 50  # Example frame
    frame_info = frame_data[frame_idx]

    # Load RGB
    rgb_static = cv2.imread(str(episode_dir / frame_info["static_top_rgb"]))
    rgb_gripper = cv2.imread(str(episode_dir / frame_info["eoat_right_top_rgb"]))

    # Load raw segmentation
    seg_static = cv2.imread(str(episode_dir / frame_info["static_top_segmentation_raw"]),
                           cv2.IMREAD_GRAYSCALE)

    return {
        'frame_data': frame_data,
        'metadata': metadata,
        'rgb_static': rgb_static,
        'rgb_gripper': rgb_gripper,
        'seg_static': seg_static,
        'state': frame_info['state'],
        'action': frame_info['action']
    }
```

### PyTorch Dataset Example

```python
import torch
from torch.utils.data import Dataset

class A1GalaxeaDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.episodes = list(self.data_root.glob("**/20*"))
        self.transform = transform

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_data = load_episode_data(self.episodes[idx])

        # Convert to tensors
        rgb_static = torch.from_numpy(episode_data['rgb_static']).permute(2,0,1)
        seg_static = torch.from_numpy(episode_data['seg_static']).long()

        if self.transform:
            rgb_static = self.transform(rgb_static)

        return {
            'rgb_static': rgb_static,
            'seg_static': seg_static,
            'state': torch.tensor(episode_data['state']),
            'action': torch.tensor(episode_data['action'])
        }
```

---

## Troubleshooting

### Common Issues

**1. Environment Setup**

```bash
# Ensure ManiSkill environment is activated
conda activate maniskill

# Verify A1 Galaxea is available
python -c "import gymnasium as gym; import mani_skill; env = gym.make('PickBox-v1', robot_uids='a1_galaxea')"
```

**2. Observation Mode**

```python
# MUST use segmentation-enabled observation mode
obs_mode='rgb+depth+segmentation'  # ✅ Correct
obs_mode='rgb+depth'               # ❌ Missing segmentation
```

**3. Memory Issues**

```python
# For large datasets, generate in batches
for batch in range(10):
    generate_pick_box_dataset(
        env=env,
        num_episodes=100,
        output_root_dir=f'./batch_{batch}',
        seed_start=batch * 100
    )
```

**4. Disk Space**

```bash
# Check disk usage
du -sh training_data/

# Clean up test data
rm -rf ./debug_data
```

---

## Advanced Usage

### Custom Segmentation Colors

```python
# Modify colors in the dataset generator
# Edit _extract_segmentation_mapping() method
predefined_colors = {
    "b5box": [0, 0, 255],      # Red for box
    "basket": [0, 255, 0],     # Green for basket
    "gripper": [255, 0, 0],    # Blue for gripper
    "table": [128, 128, 128],  # Gray for table
    "background": [0, 0, 0],   # Black for background
}
```

### Batch Processing

```bash
# Generate multiple datasets in parallel
python -c "
import gymnasium as gym
import mani_skill
from mani_skill.examples.motionplanning.a1_galaxea.solutions.pick_box_dataset_generator import generate_pick_box_dataset

for i in range(5):
    env = gym.make('PickBox-v1', robot_uids='a1_galaxea', obs_mode='rgb+depth+segmentation')
    generate_pick_box_dataset(env, num_episodes=20, seed_start=i*100, output_root_dir=f'./dataset_{i}')
    env.close()
"
```

---

## Summary

The A1 Galaxea dataset generator provides a complete solution for creating rich, multi-modal training datasets with:

- ✅ **Frame-by-frame capture** (every control step)
- ✅ **Dual camera perspectives** (static + gripper)
- ✅ **Multi-modal data** (RGB, depth, segmentation)
- ✅ **Video recording** (default camera + compiled MP4)
- ✅ **Training-ready format** (raw PNG segmentation with IDs 0-5)
- ✅ **Comprehensive metadata** (entity mappings, color legends)
- ✅ **Predetermined colors** (consistent visualization)

Perfect for training vision-based manipulation policies, semantic segmentation models, and multi-modal learning systems!
