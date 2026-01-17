from setuptools import find_packages, setup

package_name = 'iros_robopro_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/driver.launch.py']),
        ('share/' + package_name + '/config', ['config/driver.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mobile',
    maintainer_email='mobile@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rc_robot_driver = iros_robopro_controller.driver_node:main',
        ],
    },
)
