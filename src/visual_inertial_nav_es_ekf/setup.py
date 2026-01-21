from setuptools import setup
import os
from glob import glob

package_name = 'visual_inertial_nav_es_ekf'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Visual-Inertial EKF Project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # executable_name = package.script_name:main_function
            'visual_detector = visual_inertial_nav_es_ekf.visual_detector:main',
            'es_ekf = visual_inertial_nav_es_ekf.es_ekf:main',
            'trajectory_monitor = visual_inertial_nav_es_ekf.trajectory_monitor:main',
            'vehicle_driver = visual_inertial_nav_es_ekf.vehicle_driver:main',
        ],
    },
)
