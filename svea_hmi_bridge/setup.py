from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'svea_hmi_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='yunhaox@kth.se',
    description='SVEA HMI Bridge for UDP communication',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'udp_bridge = svea_hmi_bridge.udp_bridge_node:main',
        ],
    },
)
