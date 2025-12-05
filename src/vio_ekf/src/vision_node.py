#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Subscribe to camera images
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10)

        # Publisher for landmarks (u, v, id)
        # Using PoseArray for simplicity:
        # x = u (pixel), y = v (pixel), z = ID (1=Red, 2=Green)
        self.publisher_ = self.create_publisher(PoseArray, '/vio/landmarks', 10)

        self.bridge = CvBridge()

        # Define HSV color ranges
        # Red can wrap around 180, so we need two ranges
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # Green range
        self.lower_green = np.array([40, 70, 50])
        self.upper_green = np.array([80, 255, 255])

        self.get_logger().info("Vision Node Started. Waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create masks
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # Detect blobs
        landmarks_msg = PoseArray()
        landmarks_msg.header = msg.header # Sync timestamps

        # Process Red (ID 1)
        self.detect_and_add(mask_red, 1.0, landmarks_msg, cv_image, (0, 0, 255))

        # Process Green (ID 2)
        self.detect_and_add(mask_green, 2.0, landmarks_msg, cv_image, (0, 255, 0))

        # Publish detections
        self.publisher_.publish(landmarks_msg)

        # Optional: Debug view (comment out if running headless without X11 forwarding)
        # cv2.imshow("Debug View", cv_image)
        # cv2.waitKey(1)

    def detect_and_add(self, mask, landmark_id, msg_array, debug_img, color):
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter noise
            if area > 500:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Create Pose observation
                    pose = Pose()
                    pose.position.x = float(cx)
                    pose.position.y = float(cy)
                    pose.position.z = float(landmark_id)

                    # We don't use orientation for point features
                    pose.orientation.w = 1.0

                    msg_array.poses.append(pose)

                    # Draw on debug image
                    cv2.circle(debug_img, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.drawContours(debug_img, [contour], 0, color, 2)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()