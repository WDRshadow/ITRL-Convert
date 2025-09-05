#!/usr/bin/env python3

import socket
import struct

def udp_receiver():
    """简单的UDP接收器用于测试"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 10086))
    
    print("UDP receiver started on port 10086")
    print("Waiting for data... (Press Ctrl+C to stop)")
    
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            
            if len(data) == 8:  # 两个float32 = 8字节
                steering, velocity = struct.unpack('ff', data)
                print(f"Received from {addr}: steering={steering:.3f} rad, velocity={velocity:.3f} m/s")
            else:
                print(f"Received unexpected data length: {len(data)} bytes")
                
    except KeyboardInterrupt:
        print("\nStopping UDP receiver...")
    finally:
        sock.close()

if __name__ == '__main__':
    udp_receiver()
