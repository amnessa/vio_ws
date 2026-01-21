"""
Range + Bearing Visual Detector (Fixed)
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import numpy as np
import math

class VisualDetector(Node):
    def __init__(self):
        super().__init__('visual_detector')
        
        # Parameters
        self.fov = 1.0472 
        self.max_range = 20.0
        self.camera_offset_x = 1.0
        
        self.bearing_stddev = 0.05
        self.range_stddev = 0.1
        
        self.landmarks = {
            1: {'pos': [6.0, 6.0, 0.5],  'name': 'Red'},
            2: {'pos': [6.0, -6.0, 0.5], 'name': 'Green'},
            3: {'pos': [10.0, 0.0, 0.5], 'name': 'Blue'},
            4: {'pos': [0.0, 8.0, 0.5],  'name': 'Yellow'}
        }
        
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PointStamped, '/visual_measurement', 10)
        self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        try:
            trans = self.tf_buffer.lookup_transform('vehicle_blue/chassis', 'world', Time())
        except tf2_ros.TransformException:
            return

        for lm_id, data in self.landmarks.items():
            lm_point = PointStamped()
            lm_point.header.frame_id = 'world'
            lm_point.point.x = data['pos'][0]
            lm_point.point.y = data['pos'][1]
            lm_point.point.z = data['pos'][2]

            try:
                lm_in_chassis = self.tf_buffer.transform(lm_point, 'vehicle_blue/chassis')
            except tf2_ros.TransformException:
                continue

            # Vector from Camera to Landmark (in Body Frame)
            dx = lm_in_chassis.point.x - self.camera_offset_x
            dy = lm_in_chassis.point.y 
            
            dist = math.sqrt(dx*dx + dy*dy)
            # Standard definition: Positive = Left
            bearing = math.atan2(dy, dx)

            if dist < self.max_range and dist > 0.5 and abs(bearing) < (self.fov / 2.0):
                meas_bearing = bearing + np.random.normal(0, self.bearing_stddev)
                meas_range = dist + np.random.normal(0, self.range_stddev)
                meas_range = max(0.1, meas_range)

                msg = PointStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'vehicle_blue/camera_frame'
                msg.point.x = float(meas_bearing)
                msg.point.y = float(meas_range)
                msg.point.z = float(lm_id)
                
                self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(VisualDetector())
    rclpy.shutdown()