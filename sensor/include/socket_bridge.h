#ifndef SOCKET_BRIDGE_H
#define SOCKET_BRIDGE_H

#include <shared_mutex>
#include <arpa/inet.h>

class SocketBridge
{
    int sockfd_;
    sockaddr_in localAddr_;

public:
    SocketBridge(const std::string &ip, int port);
    ~SocketBridge();
    void discard();
    [[nodiscard]] bool isValid() const;
    ssize_t receiveData(char *buffer, size_t bufferSize) const;
};

void receive_data_loop(const SocketBridge *bridge, char *buffer, size_t bufferSize,
                       std::shared_mutex &bufferMutex);

#endif // SOCKET_BRIDGE_H
