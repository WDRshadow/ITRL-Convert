from launch import LaunchDescription
from launch_ros.actions import Node
import math


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='svea_hmi_bridge',
            executable='udp_bridge',
            name='udp_bridge_node',
            parameters=[{
                'udp_port': 10086,
                'udp_host': '0.0.0.0',
                'steering_coefficient': math.pi/4/127.0,
            }],
            output='screen'
        )
    ])
