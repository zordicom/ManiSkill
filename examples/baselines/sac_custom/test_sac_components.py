#!/usr/bin/env python3
"""
Test script for SAC custom components.
"""

import numpy as np
import torch
from multimodal_encoders import MultimodalEncoder


def test_multimodal_encoder():
    """Test the multimodal encoder."""
    print("Testing MultimodalEncoder...")

    # Create encoder
    encoder = MultimodalEncoder(
        state_dim=25,  # Typical for PickCube-v1
        image_channels=3,
        image_size=(64, 64),
        output_dim=512,
        use_dinov2=False,  # Use simple CNN for testing
    )

    # Create mock observations
    batch_size = 4
    obs = {
        "state": torch.randn(batch_size, 25),
        "rgb": torch.randint(0, 255, (batch_size, 64, 64, 3), dtype=torch.uint8),
    }

    # Test forward pass
    output = encoder(obs)
    print(f"✅ Encoder output shape: {output.shape}")
    assert output.shape == (batch_size, 512), f"Expected (4, 512), got {output.shape}"

    # Test with different image format [B, C, H, W]
    obs_chw = {
        "state": torch.randn(batch_size, 25),
        "rgb": torch.randint(0, 255, (batch_size, 3, 64, 64), dtype=torch.uint8),
    }
    output_chw = encoder(obs_chw)
    print(f"✅ Encoder output shape (CHW): {output_chw.shape}")
    assert output_chw.shape == (batch_size, 512)


def test_sac_networks():
    """Test SAC policy and Q networks."""
    print("\nTesting SAC Networks...")

    # Mock environment specs
    class MockEnvs:
        class SingleActionSpace:
            shape = (8,)  # Typical for PickCube-v1
            high = np.ones(8)
            low = -np.ones(8)

        single_action_space = SingleActionSpace()

    envs = MockEnvs()

    # Create encoder
    encoder = MultimodalEncoder(
        state_dim=25,
        image_channels=3,
        image_size=(64, 64),
        output_dim=512,
        use_dinov2=False,
    )

    # Import SAC networks here to avoid gymnasium import issues
    from sac_networks import create_sac_networks

    # Create networks using factory function
    action_high = envs.single_action_space.high
    action_low = envs.single_action_space.low
    action_dim = len(action_high)

    policy, q1, q2 = create_sac_networks(
        state_dim=25, action_dim=action_dim, action_high=action_high, action_low=action_low, use_dinov2=False
    )

    # Test forward passes
    batch_size = 4
    obs = {
        "state": torch.randn(batch_size, 25),
        "rgb": torch.randint(0, 255, (batch_size, 64, 64, 3), dtype=torch.uint8),
    }

    # Test policy
    mean, log_std = policy.forward(obs)
    print(f"✅ Policy mean shape: {mean.shape}")
    print(f"✅ Policy log_std shape: {log_std.shape}")
    assert mean.shape == (batch_size, 8)
    assert log_std.shape == (batch_size, 8)

    # Test action sampling
    action, log_prob, _ = policy.get_action(obs)
    print(f"✅ Sampled action shape: {action.shape}")
    print(f"✅ Log prob shape: {log_prob.shape}")
    assert action.shape == (batch_size, 8)
    assert log_prob.shape == (batch_size, 1)

    # Test Q-network
    q_values = q1(obs, action)
    print(f"✅ Q-values shape: {q_values.shape}")
    assert q_values.shape == (batch_size,)

    print("✅ All SAC network tests passed!")


if __name__ == "__main__":
    test_multimodal_encoder()
    test_sac_networks()
    print("\n🎉 All tests passed! The SAC custom implementation is ready to use.")
    print("\nTo run training:")
    print("python sac_custom.py --env_id='PickCube-v1' --obs_mode='rgb' \\")
    print("  --num_envs=32 --utd=0.5 --buffer_size=300_000 \\")
    print("  --control-mode='pd_ee_delta_pos' --camera_width=64 --camera_height=64 \\")
    print("  --total_timesteps=1_000_000 --eval_freq=10_000 --use_dinov2=False")
