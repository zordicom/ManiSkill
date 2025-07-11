#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Simple test script for the SAC Delta Action server.

This script demonstrates how to:
1. Start the SAC Delta Action server
2. Send test requests to verify functionality

Usage:
    # First, start the server (in another terminal):
    python playground/rl/simple_rl/serve_delta_action.py \
        --checkpoint-path path/to/sac_checkpoint_best.pt \
        --sac-config playground/rl/simple_rl/rl_galaxea_sac.yaml

    # Then run this test:
    python playground/rl/simple_rl/test_serve_delta_action.py
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import websockets


async def test_sac_delta_server(host: str = "localhost", port: int = 10014):
    """Test the SAC Delta Action server with sample data."""
    uri = f"ws://{host}:{port}"

    try:
        async with websockets.connect(uri) as websocket:
            logging.info(f"Connected to SAC Delta Action server at {uri}")

            # Test 1: Get metadata
            metadata_request = {"type": "metadata"}
            await websocket.send(json.dumps(metadata_request))
            metadata_response = await websocket.recv()
            metadata = json.loads(metadata_response)

            logging.info("Server metadata received:")
            logging.info(
                f"  Server name: {metadata.get('server_info', {}).get('server_name')}"
            )
            logging.info(
                f"  Model ID: {metadata.get('service_metadata', {}).get('model_id')}"
            )
            logging.info(
                f"  Horizon: {metadata.get('service_metadata', {}).get('horizon')}"
            )
            logging.info(
                f"  n_obs_steps: {metadata.get('service_metadata', {}).get('n_obs_steps')}"
            )

            # Test 2: Send a prediction request with dummy data
            n_obs_steps = metadata.get("service_metadata", {}).get("n_obs_steps", 11)
            state_dim = 28  # Standard joint state dimension

            # Create dummy joint states (temporal sequence)
            dummy_joint_states = (
                np.random.randn(n_obs_steps, state_dim).astype(np.float32).tolist()
            )

            # Create dummy camera images (empty for now - in real usage these would be JPEG bytes)
            camera_keys = ["eoat_left_top", "eoat_right_top", "static_top"]
            dummy_images = {}
            for key in camera_keys:
                # In real usage, this would be JPEG-encoded image bytes
                # For testing, we'll skip images or use empty bytes
                dummy_images[key] = b""  # Empty bytes - server should handle gracefully

            prediction_request = {
                "type": "predict",
                "observations": {"joint_states": dummy_joint_states, **dummy_images},
            }

            logging.info("Sending prediction request...")
            await websocket.send(
                json.dumps(
                    prediction_request,
                    default=lambda x: x.decode() if isinstance(x, bytes) else x,
                )
            )

            prediction_response = await websocket.recv()
            response = json.loads(prediction_response)

            if "error" in response:
                logging.error(f"Prediction failed: {response['error']}")
            else:
                logging.info("Prediction successful!")
                actions = response.get("actions", {})
                expert_action = actions.get("expert_action", [])
                residual_action = actions.get("residual_action", [])
                final_action = actions.get("final_action", [])

                logging.info(
                    f"  Expert action shape: {len(expert_action) if expert_action else 0}"
                )
                logging.info(
                    f"  Residual action shape: {len(residual_action) if residual_action else 0}"
                )
                logging.info(
                    f"  Final action shape: {len(final_action) if final_action else 0}"
                )

                if expert_action:
                    logging.info(
                        f"  Expert action range: [{min(expert_action):.3f}, {max(expert_action):.3f}]"
                    )
                if residual_action:
                    logging.info(
                        f"  Residual action range: [{min(residual_action):.3f}, {max(residual_action):.3f}]"
                    )
                if final_action:
                    logging.info(
                        f"  Final action range: [{min(final_action):.3f}, {max(final_action):.3f}]"
                    )

    except ConnectionRefusedError:
        logging.error(f"Could not connect to server at {uri}")
        logging.error("Make sure the server is running:")
        logging.error("  python playground/rl/simple_rl/serve_delta_action.py \\")
        logging.error("    --checkpoint-path path/to/checkpoint.pt \\")
        logging.error("    --sac-config playground/rl/simple_rl/rl_galaxea_sac.yaml")
    except Exception as e:
        logging.error(f"Test failed: {e}")


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logging.info("Testing SAC Delta Action server...")
    asyncio.run(test_sac_delta_server())


if __name__ == "__main__":
    main()
