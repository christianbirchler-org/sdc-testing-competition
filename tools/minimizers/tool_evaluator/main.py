#!python

import grpc

import interface_2027_pb2 as interface
import interface_2027_pb2_grpc as server_client



if __name__ == "__main__":
    print("\nstart tool evaluator")

    channel = grpc.insecure_channel("localhost:50051")

    tool = server_client.CompetitionToolStub(channel=channel)

    tool_name_reply: interface.NameReply = tool.Name(interface.Empty())

    print("name of the tool: {}".format(tool_name_reply.name))



