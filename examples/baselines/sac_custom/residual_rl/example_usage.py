#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Example usage of the SAC Delta Action server and client.
Demonstrates both modes: with SAC model and without SAC model.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ws_rl_client import ZordiRLClient


def create_mock_observation(
    n_obs_steps: int = 11, camera_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a mock observation for testing with structured dictionaries."""
    if camera_keys is None:
        camera_keys = ["static_top_rgb", "eoat_left_top_rgb", "eoat_right_top_rgb"]

    # Create mock state sequence as list of dictionaries
    state_sequence = []
    for i in range(n_obs_steps):
        state_dict = {
            "right_arm_joints": np.random.uniform(-1, 1, 7).tolist(),
            "right_arm_tool_pose": np.random.uniform(-1, 1, 7).tolist(),
            "pick_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "place_target_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        state_sequence.append(state_dict)

    observation = {
        "state_sequence": state_sequence,
    }

    # Create proper mock JPEG image data for each camera
    # Generate a simple colored image and encode it as JPEG

    # Create a 224x224 RGB image with random colors for each camera
    for i, camera_key in enumerate(camera_keys):
        # Create a solid color image with different colors for each camera
        color = [
            (100, 150, 200),  # Blue-ish for static_top_rgb
            (150, 100, 200),  # Purple-ish for eoat_left_top_rgb
            (200, 150, 100),  # Orange-ish for eoat_right_top_rgb
        ][i % 3]

        # Create 224x224 image
        img = np.full((224, 224, 3), color, dtype=np.uint8)

        # Add some simple pattern to make it more realistic
        cv2.rectangle(img, (50, 50), (174, 174), (255, 255, 255), 2)
        cv2.putText(
            img,
            camera_key[:8],
            (60, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Encode as JPEG
        success, encoded_img = cv2.imencode(".jpg", img)
        if success:
            observation[camera_key] = encoded_img.tobytes()
        else:
            # Fallback: create a minimal valid JPEG header
            observation[camera_key] = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\x00\xff\xd9"
            )

    return observation


async def test_client_connection(host: str = "localhost", port: int = 10012):
    """Test client connection to server."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = ZordiRLClient(host, port)

    logger.info(f"🔌 Connecting to server at {host}:{port}")

    if not client.connect_and_initialize():
        logger.error("❌ Failed to connect to server")
        return

    logger.info("✅ Connected successfully!")

    # Log server information
    client.log_server_info()

    # Test prediction with mock observation
    logger.info("🧪 Testing prediction with mock observation...")

    # Create mock observation
    mock_obs = create_mock_observation()

    # Get action breakdown
    result = client.get_action_breakdown(mock_obs)
    if result:
        expert_action, residual_action, final_action = result
        logger.info("✅ Received action breakdown:")
        logger.info(f"  Expert action: {expert_action}")
        logger.info(f"  Residual action: {residual_action}")
        logger.info(f"  Final action: {final_action}")

        # Check if residual action is zeros (base policy only mode)
        def is_close_to_zero(value):
            """Check if a value (or nested structure) is close to zero."""
            if isinstance(value, dict):
                return all(is_close_to_zero(v) for v in value.values())
            elif isinstance(value, (list, tuple)):
                return all(is_close_to_zero(v) for v in value)
            else:
                try:
                    return abs(float(value)) < 1e-6
                except (TypeError, ValueError):
                    return False

        if is_close_to_zero(residual_action):
            logger.info("  🔍 Residual action is all zeros (base policy only mode)")
        else:
            logger.info(
                "  🔍 Residual action has non-zero values (SAC + base policy mode)"
            )

        # Demonstrate field extraction
        logger.info("\n📊 Field-specific action extraction:")
        if "right_arm_joints" in final_action:
            right_arm_joints = final_action["right_arm_joints"]
            logger.info(f"  Right arm joints: {right_arm_joints}")
        if "right_arm_tool_pose" in final_action:
            right_arm_tool_pose = final_action["right_arm_tool_pose"]
            logger.info(f"  Right arm tool pose: {right_arm_tool_pose}")
    else:
        logger.error("❌ Failed to get action breakdown")

    # Test simple action getting
    logger.info("\n🧪 Testing simple action getting...")
    final_action = client.get_action_only(mock_obs)
    if final_action:
        logger.info(
            f"✅ Received final action dictionary with fields: {list(final_action.keys())}"
        )

        # Test the helper method for field extraction
        right_arm_joints = client.get_action_field_value(
            final_action, "right_arm_joints"
        )
        if right_arm_joints:
            logger.info(f"  Right arm joints extracted: {right_arm_joints}")
    else:
        logger.error("❌ Failed to get final action")

    client.close()
    logger.info("👋 Disconnected from server")


def run_server_examples():
    """Print example commands for running the server."""
    print("=" * 60)
    print("📖 SERVER USAGE EXAMPLES")
    print("=" * 60)

    print("\n🚀 Run with SAC model (full functionality):")
    print("python playground/rl/residual_rl/serve_delta_action.py \\")
    print(
        "  --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \\"
    )
    print("  --sac-config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml \\")
    print(
        "  --sac-checkpoint outputs/sac_box_pnp/model_20250101_120000/checkpoint_best.pt \\"
    )
    print("  --host 0.0.0.0 --port 10012 --action-offset 0")

    print("\n🏃 Run without SAC model (base policy only):")
    print("python playground/rl/residual_rl/serve_delta_action.py \\")
    print(
        "  --expert-config config/galaxea/box_pnp_25hz/galaxea_act_25hz_modular.yaml \\"
    )
    print("  --host 0.0.0.0 --port 10012 --action-offset 0")

    print("\n⚙️ Action Offset Parameter:")
    print("  --action-offset 0   # Use first action from action chunk (default)")
    print("  --action-offset 13  # Use 13th action from action chunk")
    print(
        "  Note: ACT models produce action chunks of size [n_action_steps, action_dim]"
    )
    print("        Action offset selects which action to use from the chunk")

    print("\n📱 CLIENT USAGE EXAMPLES")
    print("=" * 60)

    print("\n🔌 Basic client connection:")
    print("from ws_rl_client import ZordiRLClient")
    print("client = ZordiRLClient('localhost', 10012)")
    print("client.connect_and_initialize()")
    print("client.log_server_info()")

    print("\n🎯 Get action from observation (NEW FORMAT):")
    print("# Observation now uses structured dictionaries:")
    print("observation = {")
    print("    'state_sequence': [")
    print("        {")
    print("            'right_arm_joints': [0.1, 0.2, ...],       # 7 values")
    print("            'right_arm_tool_pose': [x, y, z, qx, qy, qz, qw],  # 7 values")
    print("            'pick_target_pose': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],")
    print("            'place_target_pose': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],")
    print("        },")
    print("        # ... more state dictionaries for history")
    print("    ],")
    print("    'static_top_rgb': jpeg_bytes,")
    print("    'eoat_left_top_rgb': jpeg_bytes,")
    print("    'eoat_right_top_rgb': jpeg_bytes,")
    print("}")
    print("final_action = client.get_action_only(observation)")
    print("print(f'Final action: {final_action}')")

    print("\n🔍 Get detailed action breakdown (NEW FORMAT):")
    print("expert, residual, final = client.get_action_breakdown(observation)")
    print("print(f'Expert action: {expert}')      # Dictionary with field names")
    print("print(f'Residual action: {residual}')  # Dictionary with field names")
    print("print(f'Final action: {final}')        # Dictionary with field names")

    print("\n🎯 Extract specific action fields:")
    print("# All actions are now dictionaries with field names")
    print("right_arm_joints = final_action['right_arm_joints']")
    print("right_arm_tool_pose = final_action['right_arm_tool_pose']")
    print("# Or use the helper method:")
    print(
        "right_arm_joints = client.get_action_field_value(final_action, 'right_arm_joints')"
    )

    print("\n📊 Field Definitions (from server metadata):")
    print("State Fields:")
    print("  - right_arm_joints: [0:7]     # Joint positions")
    print("  - right_arm_tool_pose: [7:14] # Tool pose (position + quaternion)")
    print("  - pick_target_pose: [14:21]   # Pick target pose")
    print("  - place_target_pose: [21:28]  # Place target pose")
    print("Action Fields:")
    print("  - right_arm_joints: [0:7]     # Joint commands")
    print("  - right_arm_tool_pose: [7:14] # Tool pose commands")
    print("  - pick_target_pose: [14:21]   # Pick target commands")
    print("  - place_target_pose: [21:28]  # Place target commands")


def run_training_examples():
    """Print example commands for training the SAC model."""
    print("=" * 60)
    print("🎓 TRAINING EXAMPLES")
    print("=" * 60)

    print("\n🚀 Train SAC delta action model:")
    print("python playground/rl/residual_rl/try_sac_delta_action.py \\")
    print("  --config playground/rl/residual_rl/rl_galaxea_sac_box_pnp.yaml")

    print("\n📊 Monitor training progress:")
    print("# Check outputs/sac_box_pnp/model_YYYYMMDD_HHMMSS/ for checkpoints")
    print("# Use checkpoint_best.pt for serving")


async def main():
    """Main function demonstrating the complete pipeline."""
    print("=" * 60)
    print("🤖 SAC DELTA ACTION - COMPLETE PIPELINE DEMO")
    print("=" * 60)

    # Show training examples
    run_training_examples()

    # Show server examples
    run_server_examples()

    # Test client connection (you can uncomment this if server is running)
    print("\n🧪 CLIENT CONNECTION TEST")
    print("=" * 60)
    print("Uncomment the line below if your server is running:")
    print("# await test_client_connection()")

    await test_client_connection()  # Uncomment if server is running


if __name__ == "__main__":
    asyncio.run(main())
