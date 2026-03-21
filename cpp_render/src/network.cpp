#include "network.hpp"
#include "state_manager.hpp"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <iostream>

#pragma comment(lib, "ws2_32.lib")

void server_thread() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in service;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = inet_addr("127.0.0.1");
    service.sin_port = htons(8080);
    bind(ListenSocket, (SOCKADDR*)&service, sizeof(service));
    listen(ListenSocket, SOMAXCONN);
    
    while (true) {
        SOCKET AcceptSocket = accept(ListenSocket, NULL, NULL);
        if (AcceptSocket == INVALID_SOCKET) break;
        std::string buffer;
        char chunk[4096];
        while (true) {
            int bytes = recv(AcceptSocket, chunk, sizeof(chunk), 0);
            if (bytes <= 0) break;
            buffer.append(chunk, bytes);
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                try {
                    global_state.update(json::parse(buffer.substr(0, pos)));
                } catch (...) {}
                buffer.erase(0, pos + 1);
            }
        }
        closesocket(AcceptSocket);
    }
    closesocket(ListenSocket);
    WSACleanup();
}
