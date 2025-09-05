#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8
from nav_msgs.msg import Odometry
import socket
import struct
import math


class UDPBridgeNode(Node):
    def __init__(self):
        super().__init__('udp_bridge_node')
        
        self.declare_parameter('udp_port', 10086)
        self.declare_parameter('udp_host', '0.0.0.0')
        self.declare_parameter('steering_coefficient', math.pi/4/127.0)
        
        self.udp_port = self.get_parameter('udp_port').get_parameter_value().integer_value
        self.udp_host = self.get_parameter('udp_host').get_parameter_value().string_value
        self.steering_coeff = self.get_parameter('steering_coefficient').get_parameter_value().double_value
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.steering_angle = 0.0
        self.velocity_x = 0.0
        
        self.steering_sub = self.create_subscription(
            Int8,
            '/lli/ctrl/steering',
            self.steering_callback,
            10
        )
        
        self.odometry_sub = self.create_subscription(
            Odometry,
            '/odometry/local',
            self.odometry_callback,
            10
        )
        
        self.timer = self.create_timer(0.1, self.send_udp_data)
        
        self.get_logger().info(f'UDP Bridge Node started, sending to {self.udp_host}:{self.udp_port}')
        self.get_logger().info(f'Steering coefficient: {self.steering_coeff}')
    
    def steering_callback(self, msg):
        self.steering_angle = float(msg.data) * self.steering_coeff
        
        self.steering_angle = max(-math.pi/4, min(math.pi/4, self.steering_angle))
        
        self.get_logger().debug(f'Received steering: {msg.data} -> {self.steering_angle:.3f} rad')
    
    def odometry_callback(self, msg):
        self.velocity_x = msg.twist.twist.linear.x
        self.get_logger().debug(f'Received velocity_x: {self.velocity_x:.3f} m/s')
    
    def send_udp_data(self):
        try:
            data = struct.pack('<ff', self.steering_angle, self.velocity_x)
            
            self.udp_socket.sendto(data, (self.udp_host, self.udp_port))
            
            self.get_logger().debug(f'Sent UDP: steering={self.steering_angle:.3f}, velocity={self.velocity_x:.3f}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to send UDP data: {str(e)}')
    
    def destroy_node(self):
        if hasattr(self, 'udp_socket'):
            self.udp_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = UDPBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
