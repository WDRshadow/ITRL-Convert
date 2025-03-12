#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>

#include "udp_server.h"

#define BROADCAST_IP "255.255.255.255"
#define PORT 10000
#define NUM_FLOATS 25

uint32_t float_to_big_endian(float f) {
    uint32_t temp;
    std::memcpy(&temp, &f, sizeof(float));
    return htonl(temp);
}

void start_server() {
    const int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    int optval = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)) < 0) {
        std::cerr << "[udp server] setsockopt(SO_REUSEPORT) failed." << std::endl;
        close(sock);
        return;
    }

    int broadcastEnable = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcastEnable, sizeof(broadcastEnable)) < 0) {
        perror("[udp server] broadcast enable failed");
        close(sock);
        exit(EXIT_FAILURE);
    }

    sockaddr_in localAddr = {};
    std::memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_port = htons(PORT);
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, reinterpret_cast<sockaddr*>(&localAddr), sizeof(localAddr)) < 0) {
        perror("[udp server] bind failed");
        close(sock);
        exit(EXIT_FAILURE);
    }

    sockaddr_in broadcastAddr = {};
    std::memset(&broadcastAddr, 0, sizeof(broadcastAddr));
    broadcastAddr.sin_family = AF_INET;
    broadcastAddr.sin_port = htons(PORT);
    broadcastAddr.sin_addr.s_addr = inet_addr(BROADCAST_IP);
    std::cout << "[udp server] Broadcasting data to " << BROADCAST_IP << std::endl;

    while (true) {
        uint32_t buffer[NUM_FLOATS];
        for (int i = 0; i < NUM_FLOATS; i++) {
            // random float values between 0 and 1
            float f = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            buffer[i] = float_to_big_endian(f);
        }

        ssize_t sentBytes = sendto(sock, buffer, sizeof(buffer), 0,
                                   reinterpret_cast<sockaddr*>(&broadcastAddr), sizeof(broadcastAddr));
        if (sentBytes < 0) {
            perror("[udp server] sendto failed");
        }

        usleep(10000); // 10 ms
    }
}
