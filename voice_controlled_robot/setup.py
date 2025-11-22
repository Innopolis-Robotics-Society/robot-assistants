from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'voice_controlled_robot'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
]

# Рекурсивно проходим по models/ и добавляем каталоги как есть
for root, dirs, files in os.walk('models'):
    if not files:
        continue
    # root уже содержит 'models/...'
    dest_dir = os.path.join('share', package_name, root)
    src_files = [os.path.join(root, f) for f in files]
    data_files.append((dest_dir, src_files))

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=data_files,
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
