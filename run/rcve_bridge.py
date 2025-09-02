#!/usr/bin/env python

import fleetmqsdk
import socket
import time
import signal
import sys
import struct
import datetime
from threading import Thread

EVERY_IP = "0.0.0.0"
UDP_IP_LOCAL = "192.168.1.121"


def main():
    fleetmq = fleetmqsdk.FleetMQ()
    config, addresses = fleetmq.getConfig(True)
    threads = []
    # sendToControlTowerThread = Thread(target=sendToControlTower, args=(fleetmq, "rcve_state"))
    # sendToControlTowerThread.start()
    # threads.append(sendToControlTowerThread)
    receiveFromControlTowerThread =Thread(target=receiveFromControlTower, args=(fleetmq, "control"))
    receiveFromControlTowerThread.start()
    threads.append(receiveFromControlTowerThread)
    metric_thread = Thread(target=pullMetrics, args=(fleetmq,))
    metric_thread.start()
    threads.append(metric_thread)

def handleExit(configReqInterface, configUpdateInterface):
    configReqInterface.close()
    configUpdateInterface.close()

def receiveFromControlTower(fleetmq, topic):
    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        data = fleetmq.receiveBytes(topic)
        if data != None:        
            # fdb = struct.unpack('<IfffII',data)
            # print(fdb) 	
            sock_tx.sendto(data, (UDP_IP_LOCAL, 10003))
        time.sleep(0.01)
        
def sendToControlTower(fleetmq, topic):
    sock_rx_rcve = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx_rcve.bind((EVERY_IP, 10002))
    while True:
        data, _ = sock_rx_rcve.recvfrom(8192)
        fleetmq.publishBytes(topic, data)
        time.sleep(0.01)

def pullMetrics(fleetmq):
    sock_tx_latency = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/latency_{timestamp}.txt"
    # with open(filename, "a") as f:
    while True:
        metrics = fleetmq.pullMetric()
        if metrics != None:
            for metric in metrics:
                print("Metric: ", metric.type, " of value ", metric.value, ", peer ", metric.peer, " at timestamp ", metric.timestamp)
            latency = int(metrics[0].value / 2)
            latency_bytes = struct.pack('<I', latency)
            sock_tx_latency.sendto(latency_bytes, (UDP_IP_LOCAL, 10088))
            now = datetime.datetime.now().isoformat(timespec="milliseconds")
            print(f"{now}: {latency} ms ")
            # f.write(f"{now},{latency}\n")
        time.sleep(0.01)

def signal_handler(sig, frame):
    print("Shutting down...")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
