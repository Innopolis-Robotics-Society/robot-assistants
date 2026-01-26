from setuptools import find_packages, setup
from glob import glob
import os

package_name = "iros_rust_detect_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mobile",
    maintainer_email="mobile@todo.todo",
    description="Rust detection node (segmentation -> bbox + overlay)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rust_detect_node = iros_rust_detect_ros.rust_detect_node:main",
        ],
    },
)
