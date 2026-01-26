from setuptools import setup

package_name = 'iros_voice_command'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Voice feedback for recognized voice commands',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'iros_voice_command_node = iros_voice_command.voice_command_node:main',
        ],
    },
)

