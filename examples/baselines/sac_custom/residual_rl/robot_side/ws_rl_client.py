"""
Copyright 2025 Zordi, Inc. All rights reserved.

WebSocket client for communicating with the SAC Delta Action policy server.

This client provides a high-level interface for robot control systems to communicate
with policy servers that combine base expert policies with optional SAC delta action
refinements. The server can operate in two modes:

1. SAC + Base Policy mode: Uses a trained SAC model to add delta corrections to
   expert actions for improved performance
2. Base Policy Only mode: Uses only the expert/base policy without SAC refinements

Key Features:
- Automatic server capability detection: Detects whether SAC model is available
- Robust observation handling: Supports multi-modal observations (state + images)
- Flexible action extraction: Provides both individual and combined action access
- Error resilience: Graceful handling of connection and data format issues
- Type-safe processing: Handles nested action structures and type conversions

Expected Input Format:
The observation dictionary should contain:
- "joint_states": List of state vectors for observation history
  Format: [[state_t-n], [state_t-n+1], ..., [state_t]]
  Each state vector contains joint positions/velocities (typically 28-dim for Galaxea)
- Camera images: JPEG-encoded bytes for each camera view
  Use cv2.imencode('.jpg', image)[1].tobytes() to encode images
  Camera names should match trained model expectations (e.g., 'static_top_rgb')

Expected Output Format:
The action response contains:
- "expert_action": Direct output from base/expert policy model
- "residual_action": Delta corrections from SAC (zeros if no SAC available)
- "final_action": Combined action ready for robot execution (expert + residual)
All actions are typically 28-dimensional for Galaxea robots.

Usage Patterns:

Basic Robot Control:
1. Initialize client: ZordiRLClient('localhost', 10012)
2. Connect: client.connect_and_initialize()
3. In control loop:
   - Prepare observation with joint_states and camera images
   - Get action: final_action = client.get_action_only(observation)
   - Execute action on robot

Advanced Usage:
- Use get_action_breakdown() to get expert, residual, and final actions separately
- Check client.has_sac to detect server mode
- Access client.server_metadata for server configuration details

Works with both SAC+Base policy mode and Base policy only mode.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from zordi_policy_rpc.direct.client import DirectClient

logger = logging.getLogger(__name__)


class ZordiRLClient(DirectClient):
    """Enhanced client for SAC Delta Action server that supports both modes."""

    def __init__(self, host: str, port: int, client_id: str | None = None):
        super().__init__(host, port)
        self.client_id = client_id or f"client_{int(time.time())}"
        self._server_metadata: Optional[Dict[str, Any]] = None
        self._has_sac: Optional[bool] = None

    def connect_and_initialize(self) -> bool:
        """Connect to server and fetch metadata for initialization."""
        try:
            self.connect()
            metadata = self.get_metadata()
            self._server_metadata = dict(metadata)  # Convert to dict for type safety

            # Extract server capabilities
            if self._server_metadata:
                service_meta = self._server_metadata.get("service_metadata", {})
                self._has_sac = service_meta.get("has_sac", False)
                model_id = service_meta.get("model_id", "unknown")

                mode = "SAC + Base Policy" if self._has_sac else "Base Policy Only"
                logger.info(f"🚀 Connected to server in {mode} mode")
                logger.info(f"Model ID: {model_id}")
                logger.info(f"Horizon: {service_meta.get('horizon', 'unknown')}")
                logger.info(
                    f"Observation steps: {service_meta.get('n_obs_steps', 'unknown')}"
                )
                logger.info(
                    f"Action steps: {service_meta.get('n_action_steps', 'unknown')}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to connect and initialize: {e}")
            return False

    @property
    def has_sac(self) -> bool:
        """Check if server has SAC model available."""
        return self._has_sac if self._has_sac is not None else False

    @property
    def server_metadata(self) -> Dict[str, Any]:
        """Get cached server metadata."""
        return self._server_metadata or {}

    # ---------------------------------------------------------------- predict
    def predict(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """
        Send observation to server and get predicted action.

        Args:
            observation: Dictionary containing observation data

        Returns:
            dict[str, Any]: Predicted action dictionary with keys:
                - expert_action: Base policy action
                - residual_action: SAC delta action (zeros if no SAC)
                - final_action: Combined action
            Returns None if error occurred
        """
        payload = {"type": "predict", "observations": observation}

        try:
            response = self.request(payload)

            # Check for error in response
            if "error" in response:
                logger.error(f"Server returned error: {response['error']}")
                return None

            # Check for actions in response
            if "actions" not in response:
                logger.error(f"Missing 'actions' field in server response: {response}")
                return None
            elif not isinstance(response["actions"], dict):
                logger.error(f"Invalid 'actions' field in server response: {response}")
                return None

            actions = response["actions"]

            # Validate expected action structure
            required_keys = ["expert_action", "residual_action", "final_action"]
            missing_keys = [key for key in required_keys if key not in actions]
            if missing_keys:
                logger.warning(f"Missing action keys: {missing_keys}")

            # Log action information in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                expert_action = actions.get("expert_action", {})
                residual_action = actions.get("residual_action", {})
                final_action = actions.get("final_action", {})

                logger.debug(f"Expert action: {expert_action}")
                logger.debug(f"Residual action: {residual_action}")
                logger.debug(f"Final action: {final_action}")

                # Check if residual action is all zeros (no SAC mode)
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

                if residual_action and is_close_to_zero(residual_action):
                    logger.debug("Residual action is zero (base policy only mode)")

            return actions

        except Exception as e:
            logger.error(f"Error during prediction request: {e}")
            return None

    # ---------------------------------------------------------------- helpers
    def test_connection(self) -> bool:
        """Test connection to server and log capabilities."""
        try:
            return self.connect_and_initialize()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_action_only(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """
        Convenient method to get only the final action as a dictionary.

        Args:
            observation: Dictionary containing observation data

        Returns:
            dict[str, Any]: Final action as a dictionary with field names, or None if error
        """
        actions = self.predict(observation)
        if actions and "final_action" in actions:
            return actions["final_action"]
        return None

    def get_action_breakdown(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        """
        Get action breakdown showing expert, residual, and final actions.

        Args:
            observation: Dictionary containing observation data

        Returns:
            tuple: (expert_action, residual_action, final_action) as dictionaries or None if error
        """
        actions = self.predict(observation)
        if actions:
            expert = actions.get("expert_action", {})
            residual = actions.get("residual_action", {})
            final = actions.get("final_action", {})

            # Ensure all actions are dictionaries
            if not isinstance(expert, dict):
                logger.warning(f"Expected expert_action to be dict, got {type(expert)}")
                expert = {}
            if not isinstance(residual, dict):
                logger.warning(
                    f"Expected residual_action to be dict, got {type(residual)}"
                )
                residual = {}
            if not isinstance(final, dict):
                logger.warning(f"Expected final_action to be dict, got {type(final)}")
                final = {}

            return expert, residual, final
        return None

    def log_server_info(self) -> None:
        """Log detailed server information."""
        if not self._server_metadata:
            logger.warning(
                "No server metadata available. Call connect_and_initialize() first."
            )
            return

        server_info = self._server_metadata.get("server_info", {})
        service_meta = self._server_metadata.get("service_metadata", {})

        logger.info("=== Server Information ===")
        logger.info(f"Server Name: {server_info.get('server_name', 'unknown')}")
        logger.info(f"Service Type: {server_info.get('service_type', 'unknown')}")
        logger.info(f"Model ID: {service_meta.get('model_id', 'unknown')}")
        logger.info(f"Has SAC: {service_meta.get('has_sac', False)}")
        logger.info(f"Horizon: {service_meta.get('horizon', 'unknown')}")
        logger.info(f"Observation Steps: {service_meta.get('n_obs_steps', 'unknown')}")
        logger.info(f"Action Steps: {service_meta.get('n_action_steps', 'unknown')}")
        logger.info(f"Action Offset: {service_meta.get('action_offset', 'unknown')}")
        logger.info(f"State Fields: {service_meta.get('state_fields', 'unknown')}")
        logger.info(f"Action Fields: {service_meta.get('action_fields', 'unknown')}")
        logger.info("==========================")

    def get_action_field_value(
        self, action_dict: dict[str, Any], field_name: str
    ) -> list[float] | None:
        """
        Extract a specific field from action dictionary.

        Args:
            action_dict: Action dictionary from server
            field_name: Name of the field to extract

        Returns:
            list[float]: Field values as a list, or None if not found
        """
        if not isinstance(action_dict, dict):
            logger.warning(f"Expected action_dict to be dict, got {type(action_dict)}")
            return None

        if field_name not in action_dict:
            logger.warning(f"Field '{field_name}' not found in action dictionary")
            return None

        field_values = action_dict[field_name]
        if isinstance(field_values, list):
            return [float(v) for v in field_values]
        else:
            logger.warning(
                f"Expected field '{field_name}' to be list, got {type(field_values)}"
            )
            return None
