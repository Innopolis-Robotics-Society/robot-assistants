from glob import glob
import os

from setuptools import find_packages, setup

package_name = "cv_session_hub"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mobile",
    maintainer_email="mobile@todo.todo",
    description="CV session hub: collects rust+pcb results and republishes last snapshot",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "cv_session_hub = cv_session_hub.cv_session_hub_node:main",
        ],
    },
)
