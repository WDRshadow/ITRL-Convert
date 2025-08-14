#include <thread>
#include <gtest/gtest.h>
#include <shared_mutex>

#include "socket_bridge.h"
#include "sensor.h"
#include "udp_server.h"

#define NUM_FLOATS 25
#define BUFFER_SIZE 8192
#define PORT 10086
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
        float f = getFloatAt(buffer, i);
        EXPECT_EQ(f, i);
    }
}

TEST(SOCKET_BRIDGE, ASYNC)
{
    std::shared_mutex bufferMutex;
    char* buffer = new char[BUFFER_SIZE];
    SocketBridge bridge(IP_ADDRESS, PORT);
    bool flag = false;
    bool isRunning = false;
    std::thread t(receive_data_loop, &bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex), std::ref(flag), std::ref(isRunning));
    sleep(2);
    for (int i = 0; i < NUM_FLOATS; i++)
    {
        std::shared_lock lock(bufferMutex);
        float f = getFloatAt(buffer, i);
        EXPECT_EQ(f, i);
    }
    flag = true;
    if (t.joinable())
    {
        int count = 0;
        while (isRunning && count++ < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (isRunning)
        {
            std::cerr << "[socket bridge] Sensor thread did not exit gracefully" << std::endl;
            t.detach();
        } else {
            t.join();
        }
    }
}

TEST(SOCKET_BRIDGE, VELOCITY)
{
    std::shared_mutex bufferMutex;
    char* buffer = new char[BUFFER_SIZE];
    SocketBridge bridge(IP_ADDRESS, PORT);
    bool flag = false;
    bool isRunning = false;
    std::thread t(receive_data_loop, &bridge, buffer, BUFFER_SIZE, std::ref(bufferMutex), std::ref(flag), std::ref(isRunning));
    const SensorAPI sensor(Velocity, buffer, BUFFER_SIZE, bufferMutex);
    sleep(2);
    EXPECT_EQ(sensor.get_float_value(), 10);
    flag = true;
    if (t.joinable())
    {
        int count = 0;
        while (isRunning && count++ < 10)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (isRunning)
        {
            std::cerr << "[socket bridge] Sensor thread did not exit gracefully" << std::endl;
            t.detach();
        } else {
            t.join();
        }
    }
}
