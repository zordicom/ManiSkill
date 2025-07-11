#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PoseStampedPublisher(Node):
    def __init__(self):
        super().__init__("pose_stamped_publisher")
        self.publisher_ = self.create_publisher(
            PoseStamped, "/a1_ee_target_follower_r", 10
        )
        self.timer = self.create_timer(
            1.0, self.timer_callback
        )  # Timer set for 1 second
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "world"
        self.pose_msg.pose.position.x = 0.14
        self.pose_msg.pose.position.y = 0.1
        self.pose_msg.pose.position.z = 0.4
        self.pose_msg.pose.orientation.x = 0.8
        self.pose_msg.pose.orientation.y = 0.24
        self.pose_msg.pose.orientation.z = 0.45
        self.pose_msg.pose.orientation.w = -0.15

    def timer_callback(self):
        self.pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.pose_msg)
        self.get_logger().info(
            "Published PoseStamped message to /a1_ee_target_follower_r"
        )
        rclpy.shutdown()  # Shutdown after publishing


def main(args=None):
    rclpy.init(args=args)
    pose_stamped_publisher = PoseStampedPublisher()
    rclpy.spin(pose_stamped_publisher)


if __name__ == "__main__":
    main()
