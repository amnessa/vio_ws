#!/usr/bin/env python3
"""
ArUco Vision Node for VIO EKF

Detects ArUco markers (DICT_4X4_50) and publishes their pixel coordinates
with unique IDs. This replaces color-based detection for robust data association.

Based on recommendation2.md:
- Unique ArUco IDs eliminate correspondence ambiguity
- High marker density ensures 3+ landmarks visible for observability
- Subpixel corner refinement improves measurement accuracy
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
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
        # x = u (pixel), y = v (pixel), z = ArUco marker ID
        self.publisher_ = self.create_publisher(PoseArray, '/vio/landmarks', 10)

        self.bridge = CvBridge()

        # ArUco Dictionary (Must match the generation script: DICT_4X4_50)
        # Using OpenCV 4.5.x API
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # === AGGRESSIVE DETECTION FOR MULTIPLE MARKERS ===
        # ES-EKF requires 3+ landmarks visible at all times for observability

        # Subpixel corner refinement for better accuracy
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 50
        self.aruco_params.cornerRefinementMinAccuracy = 0.01

        # Adaptive thresholding - wider range for varying lighting/distances
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 153  # Much larger for distant markers
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7

        # Marker size detection - VERY permissive to catch all sizes
        self.aruco_params.minMarkerPerimeterRate = 0.005  # Detect tiny markers
        self.aruco_params.maxMarkerPerimeterRate = 8.0    # And very large ones

        # Relaxed contour detection
        self.aruco_params.polygonalApproxAccuracyRate = 0.08  # More tolerant of imperfect squares
        self.aruco_params.minCornerDistanceRate = 0.01
        self.aruco_params.minDistanceToBorder = 1
        self.aruco_params.minMarkerDistanceRate = 0.005  # Allow very close markers

        # Bit extraction - CRITICAL for angled/distant markers
        self.aruco_params.perspectiveRemovePixelPerCell = 4  # Lower = more tolerant
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.3  # Larger margin
        self.aruco_params.maxErroneousBitsInBorderRate = 0.8  # Very tolerant of border issues
        self.aruco_params.errorCorrectionRate = 1.0  # Maximum error correction

        # Detection statistics
        self.detection_count = 0
        self.last_log_time = self.get_clock().now()
        self.debug_save_counter = 0

        # CLAHE for contrast enhancement (helps with washed-out simulation textures)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.get_logger().info("ArUco Vision Node Started (DICT_4X4_50, 24 markers in circular pattern)")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to enhance contrast (simulation textures often have poor contrast)
        gray = self.clahe.apply(gray)

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        landmarks_msg = PoseArray()
        landmarks_msg.header = msg.header

        # Debug: save image with detections every 100 frames
        self.detection_count += 1
        if self.detection_count % 100 == 1:
            debug_img = cv_image.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
            # Also draw rejected candidates in red
            if len(rejected) > 0:
                for rej in rejected:
                    pts = rej[0].astype(np.int32)
                    cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)

            save_path = f"/tmp/aruco_debug_{self.debug_save_counter}.png"
            cv2.imwrite(save_path, debug_img)
            self.get_logger().info(
                f"DEBUG: Saved {save_path} - Detected: {len(ids) if ids is not None else 0}, "
                f"Rejected: {len(rejected)}"
            )
            self.debug_save_counter += 1

        if ids is not None:
            for i in range(len(ids)):
                # Get center of the marker (u, v) from corner average
                c = corners[i][0]
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                marker_id = float(ids[i][0])

                # Create Pose observation
                pose = Pose()
                pose.position.x = cx  # u pixel coordinate
                pose.position.y = cy  # v pixel coordinate
                pose.position.z = marker_id  # ArUco ID for data association
                pose.orientation.w = 1.0

                landmarks_msg.poses.append(pose)

            # Log detection with marker IDs
            self.get_logger().info(
                f"ArUco DETECTED: {len(ids)} markers, IDs={ids.flatten().tolist()}, "
                f"positions={(cx, cy)}",
                throttle_duration_sec=1.0
            )

            # Debug visualization (uncomment for debugging)
            # cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            # cv2.imshow("ArUco View", cv_image)
            # cv2.waitKey(1)

        # Always publish (empty if no markers detected)
        self.publisher_.publish(landmarks_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()