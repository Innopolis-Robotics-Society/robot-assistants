from setuptools import setup

package_name = 'iros_text_to_speach'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'std_msgs',
        'sound_play',
        'kokoro>=0.9.2',  # модель из HuggingFace
        ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Voice feedback for recognized voice commands',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'voice_command_node = iros_text_to_speach.voice_command_node:main',
        ],
    },
)

