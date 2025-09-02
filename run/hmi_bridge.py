import socket
import threading
import signal
import sys
import time
import queue
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='[hmi-bridge] %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

EVERY_IP = '0.0.0.0'
RCVE_IP = '127.0.0.1'
RCVE_SUB_IP = '192.168.1.20'

RCVE_SUB_PORT = 10000
RCVE_VIDEO_PORT = 10086
RCVE_FLEETMQ_PORT = 10002

RCVE_SUB_PORT_2 = 10001
RCVE_VIDEO_PORT_2 = 10087
RCVE_FLEETMQ_PORT_2 = 10003

FORWARDING_CONFIG = [
    {
        'listen_ip': EVERY_IP,
        'listen_port': RCVE_SUB_PORT,
        'target_ip': [RCVE_IP],
        'target_port': [RCVE_VIDEO_PORT],
        'delay_ms': 0  # customize delay in milliseconds
    },
    {
        'listen_ip': EVERY_IP,
        'listen_port': RCVE_FLEETMQ_PORT_2,
        'target_ip': [RCVE_IP, RCVE_IP],
        'target_port': [RCVE_SUB_PORT_2, RCVE_VIDEO_PORT_2],
        'delay_ms': 0  # fixed
    },
    {
        'listen_ip': EVERY_IP,
        'listen_port': RCVE_SUB_PORT_2,
        'target_ip': [RCVE_SUB_IP],
        'target_port': [RCVE_SUB_PORT_2],
        'delay_ms': 0 # customize delay in milliseconds
    }
]

def udp_forwarder(listen_ip: str, listen_port: int, target_ip: list, target_port: list, delay_ms: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        logger.warning("SO_REUSEPORT not supported")

    sock.bind((listen_ip, listen_port))
    
    targets = []
    for i in range(len(target_ip)):
        targets.append(f"{target_ip[i]}:{target_port[i]}")

    logger.info(f"Listening {listen_ip}:{listen_port}, transfer to {targets} with delay {delay_ms}ms")

    packet_queue = queue.Queue()
    
    def sender_thread():
        while True:
            try:
                data, recv_time, targets_info = packet_queue.get(timeout=1.0)
                
                send_time = recv_time + (delay_ms / 1000.0)
                current_time = time.time()
                
                if current_time < send_time:
                    time.sleep(send_time - current_time)
                
                for i in range(len(target_ip)):
                    sock.sendto(data, (target_ip[i], target_port[i]))
                    
                packet_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Sender thread error: {e}")
    
    if delay_ms > 0:
        sender = threading.Thread(target=sender_thread, daemon=True)
        sender.start()

    while True:
        try:
            data, _ = sock.recvfrom(8192)
            recv_time = time.time()
            
            if delay_ms > 0:
                packet_queue.put((data, recv_time, targets))
            else:
                for i in range(len(target_ip)):
                    sock.sendto(data, (target_ip[i], target_port[i]))
                    
        except Exception as e:
            logger.error(f"Receiver error: {e}")
            break

def signal_handler(sig, frame):
    logger.info("Shutting down...")
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UDP Forwarder with customizable delay')
    parser.add_argument('-delay', type=int, default=0, metavar='<num_ms>',
                       help='Delay in milliseconds for packet forwarding (default: 0)')
    args = parser.parse_args()
    FORWARDING_CONFIG[0]['delay_ms'] = args.delay
    FORWARDING_CONFIG[2]['delay_ms'] = args.delay
    
    signal.signal(signal.SIGINT, signal_handler)
    threads = []
    for config in FORWARDING_CONFIG:
        t = threading.Thread(
            target=udp_forwarder,
            args=(
                config['listen_ip'],
                config['listen_port'],
                config['target_ip'],
                config['target_port'],
                config['delay_ms']
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
