# SAC Delta Action Pipeline - Updated for Structured Rollout Data

This document describes the updates made to support the new structured rollout data format introduced with the `ws_rl_runner.py` and `serve_delta_action.py` integration.

## What Changed

### 1. **Data Format Updates**

- **States and Actions**: Now stored as structured dictionaries with field names instead of flat vectors
- **Field Definitions**: Metadata includes `state_fields` and `action_fields` mappings for conversion
- **Extra Observations**: New `extra_obs` field contains additional axis information for reward computation
- **Depth Images**: Depth images are now collected but ignored during training (future exploration)

### 2. **Pipeline Updates**

#### **rl_dataset.py**

- Added structured data parsing using field definitions from metadata
- Implemented `_convert_state_dict_to_vector()` and `_convert_action_dict_to_vector()` methods
- Added support for extra observations in the observation space
- Enhanced validation and testing capabilities

#### **try_sac_delta_action.py**

- Updated `MultimodalEncoder` to handle extra observation fields
- Added configurable encoders for axis information (`ee_x_axis`, `ee_z_axis`, `target_x_axis`)
- Maintained compatibility with existing serving infrastructure

#### **rl_configs.py**

- Added configuration options for extra observation processing
- Added structured data processing settings
- Enhanced network configuration for multimodal observations

#### **rl_galaxea_sac_box_pnp.yaml**

- Updated dataset path and comments for new format
- Configured for structured state/action processing
- Set to ignore depth images initially

## Data Flow

### 1. **Rollout Collection** (`ws_rl_runner.py` + `serve_delta_action.py`)

```
Robot State → Structured Dictionary → Rollout File
{
  "right_arm_joints": [7 values],
  "right_arm_tool_pose": [7 values],
  "pick_target_pose": [7 values],
  "place_target_pose": [7 values]
}
```

### 2. **Dataset Loading** (`rl_dataset.py`)

```
Structured Data → Field Definitions → Flat Vectors → Model Input
Dictionary + Metadata → Convert using field mappings → Training data
```

### 3. **SAC Training** (`try_sac_delta_action.py`)

```
Multimodal Observations → SAC Networks → Delta Actions → Final Actions
[state, expert_action, images, extra_obs] → Policy/Q-networks → Residual corrections
```

### 4. **Serving** (`serve_delta_action.py`)

```
Structured Input → Flat Processing → Structured Output → Robot Execution
Client requests → Model inference → Action dictionaries → Robot control
```

## Usage Instructions

### 1. **Collect Rollouts**

```bash
# Start serving with base policy + SAC delta (or base policy only)
python playground/rl/residual_rl/serve_delta_action.py \
    --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \
    --sac-config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
    --sac-checkpoint path/to/checkpoint.pt  # optional

# Collect rollouts
python playground/rl/residual_rl/robot_side/ws_rl_runner.py \
    --record --record-dir playground/rl/residual_rl/galaxea_rollouts/box_pnp
```

### 2. **Validate Pipeline**

```bash
# Test the updated pipeline with collected data
python playground/rl/residual_rl/validate_new_format.py \
    --config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
    --rollout-dir playground/rl/residual_rl/galaxea_rollouts/box_pnp
```

### 3. **Train SAC Model**

```bash
# Train SAC delta action model on structured data
python playground/rl/residual_rl/try_sac_delta_action.py \
    --config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml
```

### 4. **Iterate**

```bash
# Use trained model for next iteration
python playground/rl/residual_rl/serve_delta_action.py \
    --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \
    --sac-config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \
    --sac-checkpoint outputs/sac_box_pnp/model_*/checkpoint_best.pt
```

## Key Features

### **Structured Data Support**

- Automatic conversion between structured dictionaries and flat vectors
- Field definitions from metadata ensure consistency
- Backward compatibility with flat vector format

### **Enhanced Observations**

- Extra observations (axis vectors) improve reward computation
- Configurable encoders for additional observation types
- Scalable architecture for future observation extensions

### **Improved Pipeline**

- End-to-end compatibility from data collection to serving
- Robust validation and testing capabilities
- Performance optimizations for training and inference

## Troubleshooting

### **Common Issues**

1. **Field Definition Mismatch**
   - Ensure `state_fields` and `action_fields` in metadata match actual data
   - Check vector dimensions against field ranges

2. **Missing Extra Observations**
   - Verify `extra_obs` is present in rollout data
   - Check that axis vectors have correct dimensions (typically 3D)

3. **Image Processing Issues**
   - Depth images are intentionally ignored - only RGB images processed
   - Ensure camera topics are available during rollout collection

### **Validation Steps**

1. Run the validation script to check data compatibility
2. Test a small training run (few epochs) to verify convergence
3. Check serving compatibility before deploying to robot

## Future Enhancements

- **Depth Image Integration**: Explore depth data for improved perception
- **Additional Extra Observations**: Support for contact forces, gripper state, etc.
- **Dynamic Field Definitions**: Runtime adaptation to different robot configurations
- **Multi-Robot Support**: Extend to dual-arm or multi-robot scenarios
