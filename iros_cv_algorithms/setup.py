from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'iros_cv_algorithms'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        "iros_cv_algorithms": ["algos/models/*.pt"],
    },
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artyom',
    maintainer_email='artyom@example.com',
    description='CV algorithms node (trigger/timer)',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv_algorithms_node = iros_cv_algorithms.cv_algorithms_node:main',
        ],
    },
)
