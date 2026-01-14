#!/usr/bin/env python3
"""
ArUco Marker Simulation Generator

This script generates:
1. 50 unique ArUco marker models for Gazebo (in ~/.gazebo/models/)
2. A new world file (landmarks.sdf) with markers arranged in a grid
3. Prints the map dictionary to paste into ekf_node.py

Based on recommendation2.md and map_upgrade.md:
- High landmark density ensures 3+ markers visible at all times
- Unique ArUco IDs eliminate data association problems
- Grid arrangement provides geometric diversity for observability

Usage:
    python3 generate_aruco_simulation.py
"""

import cv2
import cv2.aruco as aruco
import os
import numpy as np

# --- CONFIGURATION ---
MARKER_COUNT = 24  # Reduced for circular arrangement
MARKER_SIZE = 0.3  # Meters (30cm) - larger for better detection
# Output to package's models directory (Ignition uses GZ_SIM_RESOURCE_PATH)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models")
WORLD_FILE = os.path.join(SCRIPT_DIR, "..", "worlds", "landmarks.sdf")

# Select ArUco Dictionary (4x4, 50 markers)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def create_model_files(marker_id):
    """Create Gazebo model files for a single ArUco marker."""
    model_dir = os.path.join(OUTPUT_DIR, f"aruco_{marker_id}")
    os.makedirs(model_dir, exist_ok=True)

    # Generate ArUco marker image (OpenCV 4.5.x API)
    # Each marker_id produces a UNIQUE pattern
    img = np.zeros((200, 200), dtype=np.uint8)
    img = aruco.drawMarker(aruco_dict, marker_id, 200, img)
    # Add white border (crucial for detection)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # Use unique filename to avoid Ignition texture cache issues
    marker_filename = f"marker_{marker_id}.png"
    marker_path = os.path.join(model_dir, marker_filename)
    cv2.imwrite(marker_path, img)
    print(f"  Created marker {marker_id} -> {marker_path}")

    # model.config
    with open(os.path.join(model_dir, "model.config"), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>aruco_{marker_id}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>ArUco Marker {marker_id}</description>
</model>""")

    # model.sdf - Use absolute path for texture (Ignition resolves this reliably)
    # Get absolute path to the marker image
    abs_marker_path = os.path.abspath(marker_path)

    with open(os.path.join(model_dir, "model.sdf"), "w") as f:
        f.write(f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="aruco_{marker_id}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{MARKER_SIZE} {MARKER_SIZE} 0.001</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <pbr>
            <metal>
              <albedo_map>{abs_marker_path}</albedo_map>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>""")


def main():
    print(f"Generating {MARKER_COUNT} ArUco markers in {OUTPUT_DIR}...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build map dictionary string and world SDF
    map_dict = "        self.map = {\n"

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
    <plugin name='gz::sim::systems::Contact' filename='gz-sim-contact-system'/>
    <plugin name='gz::sim::systems::Imu' filename='gz-sim-imu-system'/>
    <plugin name='gz::sim::systems::Sensors' filename='gz-sim-sensors-system'>
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- Light -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include the TurtleBot3 model -->
    <include>
      <uri>model://turtlebot3_waffle_pi</uri>
      <name>turtlebot3</name>
      <pose>0 0 0.01 0 0 0</pose>
    </include>

    <!-- ArUco Marker Landmarks (Circular arrangement facing center) -->
"""

    # Place markers in concentric circles around origin
    # This ensures the robot always sees multiple markers regardless of heading
    # Ring 1: 8 markers at radius 2m (close, always visible)
    # Ring 2: 8 markers at radius 4m
    # Ring 3: 8 markers at radius 6m

    marker_id = 0
    rings = [
        (2.0, 8),   # radius, count
        (4.0, 8),
        (6.0, 8),
    ]

    for radius, count in rings:
        for j in range(count):
            create_model_files(marker_id)

            # Position on circle
            angle = 2 * np.pi * j / count
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.4  # Camera height is ~0.09m, markers slightly higher

            # Orientation: marker should face toward origin (toward camera)
            #
            # The box geometry has texture on the +Z face (top face).
            # We want this face to point TOWARD the origin.
            #
            # For a marker at position (x, y, z), the direction TO origin is (-x, -y, 0).
            # We need the +Z axis of the marker to point in direction (-x, -y, 0).
            #
            # Strategy:
            # 1. First rotate around Y by +90° (pitch) to make +Z point toward -X
            # 2. Then rotate around Z (yaw) to point toward the origin
            #
            # Yaw angle: the marker is at angle 'angle' from +X axis,
            # so to face origin, it needs to rotate by 'angle' around Z

            yaw = angle  # Rotate to face inward
            pitch = np.pi / 2  # Tilt forward so +Z faces horizontally
            roll = 0

            sdf_content += f"""
    <include>
      <uri>model://aruco_{marker_id}</uri>
      <name>landmark_{marker_id}</name>
      <pose>{x:.2f} {y:.2f} {z:.2f} {roll:.4f} {pitch:.4f} {yaw:.4f}</pose>
    </include>
"""
            map_dict += f"            {float(marker_id)}: np.array([{x:.2f}, {y:.2f}, {z:.2f}]),\n"
            marker_id += 1

    sdf_content += """
  </world>
</sdf>
"""
    map_dict += "        }"

    # Save world file
    with open(WORLD_FILE, "w") as f:
        f.write(sdf_content)

    print(f"✓ Created {MARKER_COUNT} ArUco marker models")
    print(f"✓ World saved to {WORLD_FILE}")
    print()
    print("=" * 60)
    print("COPY THIS MAP INTO YOUR ekf_node.py __init__ method:")
    print("=" * 60)
    print(map_dict)
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Paste the map dictionary into ekf_node.py")
    print("2. Rebuild: colcon build --symlink-install")
    print("3. Source: source install/setup.bash")
    print("4. Launch: ros2 launch vio_ekf vio_ekf.launch.py")


if __name__ == '__main__':
    main()
