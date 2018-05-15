from socket import *
import struct
from src.utils import whole_pb2


class AIEnv:
    BUFFER_SIZE = 1024 * 1024

    def __init__(self, ip, port):
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.sock.bind((ip, port))
        self.sock.listen(5)
        self.wait_connection()

    def recv(self):
        data = self.complete_recv(self.tcpClientSock)
        dataid, data = self.parse(data)
        return dataid, data

    def send(self, commandid, command):
        self.tcpClientSock.send(self.compose(commandid, command))

    def wait_connection(self):
        print('waiting for new connection')
        self.tcpClientSock, addr = self.sock.accept()
        print('connected from', addr)

    def complete_recv(self, sock):
        totaldata = sock.recv(self.BUFFER_SIZE)
        slen = struct.unpack('<i', totaldata[0:4])[0]
        # adjusting
        while len(totaldata) < slen:
            data = sock.recv(self.BUFFER_SIZE)
            totaldata = totaldata + data
        return totaldata

    def parse(self, data):
        stype = struct.unpack('<h', data[4:6])[0]
        newData = data[6:]
        if stype == 1:
            returnData = whole_pb2.AI_GameStartInfo()
            returnData.ParseFromString(newData)
        elif stype == 2:
            returnData = whole_pb2.AI_Frame()
            returnData.ParseFromString(newData)
        elif stype == 3:
            returnData = whole_pb2.AI_GameEnd()
            returnData.ParseFromString(newData)
        elif stype == 4:
            returnData = whole_pb2.AI_GameReadyInfo()
            returnData.ParseFromString(newData)
            if returnData.GameReady:
                print('Client ready')
            else:
                print('error in exe')
                input()
        else:
            print("No Matching ID When Recv Data")

        return stype, returnData

    def compose(self, commandid, command):
        if commandid == 100 or commandid == 105 or commandid == 106:
            commandstr = b""
        else:
            commandstr = command.SerializeToString()
        slen = len(commandstr) + 6
        return struct.pack('<i', slen) + struct.pack('<h', commandid) + commandstr
