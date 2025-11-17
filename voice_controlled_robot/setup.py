from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'voice_controlled_robot'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'resources'), 
         glob('voice_controlled_robot/resources/**/*', recursive=True)),
    ],
    install_requires=[
        'setuptools',
        'sounddevice>=0.4.6',
        'vosk>=0.3.45',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='Voice controlled UR robot using ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_controller = voice_controlled_robot.voice_controller:main',
        ],
    },
)
