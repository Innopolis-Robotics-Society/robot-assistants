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
        ('share/' + package_name + '/launch', 
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'vosk',
        'sounddevice',
        'numpy',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your-email@example.com',
    description='Voice controlled robot using Vosk speech recognition',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_controller = voice_controlled_robot.nodes.voice_controller:main',
        ],
    },
)
