#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <shared_mutex>

#include "socket_bridge.h"

SocketBridge::SocketBridge(const std::string &ip, int port)
    : sockfd_(-1), localAddr_()
{
    sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0)
    {
        std::cerr << "[socket bridge] Cannot create socket." << std::endl;
        return;
    }

    int optval = 1;
    if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval)) < 0)
    {
        std::cerr << "[socket bridge] setsockopt(SO_REUSEPORT) failed." << std::endl;
        close(sockfd_);
        sockfd_ = -1;
        return;
    }

    std::memset(&localAddr_, 0, sizeof(localAddr_));
    localAddr_.sin_family = AF_INET;
    localAddr_.sin_addr.s_addr = inet_addr(ip.c_str());
    localAddr_.sin_port = htons(port);

    if (bind(sockfd_, reinterpret_cast<const sockaddr *>(&localAddr_), sizeof(localAddr_)) < 0)
    {
        std::cerr << "[socket bridge] Port bind failed." << std::endl;
        close(sockfd_);
        sockfd_ = -1;
        return;
    }
}

SocketBridge::~SocketBridge()
{
    if (sockfd_ != -1)
    {
        close(sockfd_);
    }
}

void SocketBridge::discard()
{
    sockfd_ = -1;
}


bool SocketBridge::isValid() const
{
    return sockfd_ != -1;
}

ssize_t SocketBridge::receiveData(char *buffer, const size_t bufferSize) const
{
    if (sockfd_ < 0)
    {
        return -1;
    }
    sockaddr_in serverAddr{};
    socklen_t senderLen = sizeof(serverAddr);
    return recvfrom(sockfd_, buffer, bufferSize, 0,
                    reinterpret_cast<sockaddr *>(&serverAddr), &senderLen);
}

void receive_data_loop(const SocketBridge *bridge, char *buffer, const size_t bufferSize,
                                    std::shared_mutex &bufferMutex)
{
    auto localBuffer = new char[bufferSize];
    while (true)
    {
        if (!bridge->isValid())
        {
            break;
        }
        bridge->receiveData(localBuffer, bufferSize);
        std::lock_guard lock(bufferMutex);
        std::memcpy(buffer, localBuffer, bufferSize);
    }
    delete[] localBuffer;
}
