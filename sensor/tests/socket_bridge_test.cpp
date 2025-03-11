#include <thread>
#include <gtest/gtest.h>
#include <shared_mutex>

#include "socket_bridge.h"
#include "sensor.h"
#include "udp_server.h"

#define NUM_FLOATS 19
#define BUFFER_SIZE 8192
#define PORT 10000
#define IP_ADDRESS "0.0.0.0"

int main(int argc, char **argv)
{
    std::thread s(start_server);
    s.detach();
    sleep(1);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(SOCKET_BRIDGE, RECEIVE)
{
    const SocketBridge socket(IP_ADDRESS, PORT);
    char* buffer = new char[BUFFER_SIZE];
    EXPECT_EQ(socket.isValid(), 1);
    std::shared_mutex bufferMutex;
    socket.receiveData(buffer, BUFFER_SIZE);
    for (int i = 0; i < NUM_FLOATS; i++)
    {
        float f = BigEndianToFloat(buffer + i * sizeof(float));
        EXPECT_EQ(f, i);
    }
}

TEST(SOCKET_BRIDGE, ASYNC)
{
    std::shared_mutex bufferMutex;
    char* buffer = new char[BUFFER_SIZE];
    const SocketBridge bridge(IP_ADDRESS, PORT);
    std::thread t(receive_data_loop, &bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex));
    t.detach();
    sleep(2);
    for (int i = 0; i < NUM_FLOATS; i++)
    {
        std::shared_lock lock(bufferMutex);
        float f = BigEndianToFloat(buffer + i * sizeof(float));
        EXPECT_EQ(f, i);
    }
}

TEST(SOCKET_BRIDGE, VELOCITY)
{
    std::shared_mutex bufferMutex;
    char* buffer = new char[BUFFER_SIZE];
    const SocketBridge bridge(IP_ADDRESS, PORT);
    std::thread t(receive_data_loop, &bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex));
    t.detach();
    sleep(2);
    const SensorAPI sensor(Velocity, buffer, BUFFER_SIZE, bufferMutex);
    EXPECT_EQ(sensor.get_value(), 10);
}
