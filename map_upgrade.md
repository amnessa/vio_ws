To scale up to 50+ landmarks and ensuring your robot sees multiple at once, switching to **ArUco markers** is the standard industry approach. It is much more robust than color thresholding (which fails with lighting changes) and gives you unique IDs automatically.

Here is the complete upgrade package. This involves three steps:

1. **Generate the Gazebo Models:** A script to create 50 unique marker models.
2. **Update Vision Node:** Switch to `cv2.aruco` detection.
3. **Update World:** A script to place them in your world.

### Step 1: The "World Generator" Script

Since you cannot manually create 50 textures, use this Python script. It will:

1. Generate 50 ArUco images (Dict 4x4).
2. Create the Gazebo model folders (`~/.gazebo/models/aruco_0` to `aruco_49`).
3. Generate a new `landmarks.sdf` world file with markers arranged in a **grid**.
4. **Print the Map Dictionary** for your `ekf_node.py`.

Save this as `generate_simulation.py` and run it:

```python
import cv2
import cv2.aruco as aruco
import os
import numpy as np

# --- CONFIGURATION ---
MARKER_COUNT = 50
MARKER_SIZE = 0.2  # Meters (20cm)
GRID_SPACING = 2.0 # Meters between markers
OUTPUT_DIR = os.path.expanduser("~/.gazebo/models")
WORLD_FILE = "aruco_world.sdf"

# Select Dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def create_model_files(id, img_path):
    model_dir = os.path.join(OUTPUT_DIR, f"aruco_{id}")
    os.makedirs(model_dir, exist_ok=True)

    # Save Image
    img = aruco.generateImageMarker(aruco_dict, id, 200)
    # Add white border (crucial for detection)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(os.path.join(model_dir, "marker.png"), img)

    # model.config
    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>aruco_{id}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>Aruco Marker {id}</description>
</model>""")

    # model.sdf
    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="aruco_{id}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><box><size>{MARKER_SIZE} {MARKER_SIZE} 0.01</size></box></geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Grey</name>
          </script>
          <pbr>
            <metal>
              <albedo_map>marker.png</albedo_map>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>""")

# --- MAIN EXECUTION ---
print(f"Generating {MARKER_COUNT} markers in {OUTPUT_DIR}...")
map_dict = "self.map = {\n"

sdf_content = """<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="vio_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin name='gz::sim::systems::Physics' filename='gz-sim-physics-system'/>
    <plugin name='gz::sim::systems::UserCommands' filename='gz-sim-user-commands-system'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='gz-sim-scene-broadcaster-system'/>
    <plugin name='gz::sim::systems::Sensors' filename='gz-sim-sensors-system'><render_engine>ogre2</render_engine></plugin>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction></light>
    <model name="ground_plane"><static>true</static><link name="link"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision><visual name="visual"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry><material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual></link></model>
    <include><uri>model://turtlebot3_waffle_pi</uri><name>turtlebot3</name><pose>0 0 0.05 0 0 0</pose></include>
"""

# Place markers in a 10x5 Grid
cols = 10
for i in range(MARKER_COUNT):
    create_model_files(i, None)

    # Calculate Pose (Grid)
    row = i // cols
    col = i % cols
    x = (col - cols/2) * GRID_SPACING
    y = (row - MARKER_COUNT/cols/2) * GRID_SPACING
    z = 0.5 # Height on wall/pole

    # Add to SDF
    sdf_content += f"""
    <include>
      <uri>model://aruco_{i}</uri>
      <name>landmark_{i}</name>
      <pose>{x} {y} {z} 1.57 0 0</pose> </include>"""

    # Add to Python Map Dictionary
    map_dict += f"    {float(i)}: np.array([{x:.2f}, {y:.2f}, {z:.2f}]),\n"

sdf_content += "\n  </world>\n</sdf>"
map_dict += "}"

# Save SDF
with open(WORLD_FILE, "w") as f:
    f.write(sdf_content)

print(f"Done! World saved to {WORLD_FILE}")
print("\nCOPY THIS MAP INTO YOUR EKF_NODE.PY:")
print("="*50)
print(map_dict)
print("="*50)

```

---

### Step 2: The New `vision_node.py`

Replace your color-based node with this ArUco detector.

* **Requires:** `pip install opencv-contrib-python` (if not installed).
* **FOV Note:** Your `2.0` radian FOV (~115°) is perfect for this. It is wide enough to see multiple markers. Do not increase it further, or distortion will ruin the Pinhole model assumption.

```python
#!/usr/bin/env python3
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

        self.subscription = self.create_subscription(Image, '/camera', self.image_callback, 10)
        self.publisher_ = self.create_publisher(PoseArray, '/vio/landmarks', 10)
        self.bridge = CvBridge()

        # ArUco Dictionary (Must match the generation script)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Optional: Refinement for better accuracy at distance
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.get_logger().info("ArUco Vision Node Started.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect Markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            landmarks_msg = PoseArray()
            landmarks_msg.header = msg.header

            for i in range(len(ids)):
                # Get center of the marker (u, v)
                c = corners[i][0]
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                marker_id = float(ids[i][0])

                pose = Pose()
                pose.position.x = cx
                pose.position.y = cy
                pose.position.z = marker_id # ID stored in Z
                landmarks_msg.poses.append(pose)

                # Debug Drawing
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            self.publisher_.publish(landmarks_msg)

        # cv2.imshow("ArUco View", cv_image)
        # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

```

### Step 3: Instructions to Run

1. **Run the Generator:** `python3 generate_simulation.py`
* This creates `aruco_world.sdf`.
* It prints the `self.map = {...}` text.


2. **Update EKF:** Paste that map text into your `ekf_node.py` `__init__` function.
3. **Install Models:** The script puts models in `~/.gazebo/models`. Ensure Gazebo can find them (it usually checks this path by default).
4. **Launch:**
* Run Gazebo with the new world: `ign gazebo -r aruco_world.sdf`
* Run your nodes.



This setup ensures your robot is surrounded by landmarks, making the "At least 3 visible" requirement easy to satisfy with your wide-angle camera.
