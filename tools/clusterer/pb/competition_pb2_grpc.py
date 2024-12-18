# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import warnings

import grpc
from pb import competition_pb2 as pb_dot_competition__pb2

GRPC_GENERATED_VERSION = "1.67.0"
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + f" but the generated code in pb/competition_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class CompetitionToolStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Name = channel.unary_unary(
            "/CompetitionTool/Name",
            request_serializer=pb_dot_competition__pb2.Empty.SerializeToString,
            response_deserializer=pb_dot_competition__pb2.NameReply.FromString,
            _registered_method=True,
        )
        self.Initialize = channel.stream_unary(
            "/CompetitionTool/Initialize",
            request_serializer=pb_dot_competition__pb2.Oracle.SerializeToString,
            response_deserializer=pb_dot_competition__pb2.InitializationReply.FromString,
            _registered_method=True,
        )
        self.Select = channel.stream_stream(
            "/CompetitionTool/Select",
            request_serializer=pb_dot_competition__pb2.SDCTestCase.SerializeToString,
            response_deserializer=pb_dot_competition__pb2.SelectionReply.FromString,
            _registered_method=True,
        )


class CompetitionToolServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Name(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Initialize(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Select(self, request_iterator, context):
        """bidirectional streaming for high flexibility"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_CompetitionToolServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Name": grpc.unary_unary_rpc_method_handler(
            servicer.Name,
            request_deserializer=pb_dot_competition__pb2.Empty.FromString,
            response_serializer=pb_dot_competition__pb2.NameReply.SerializeToString,
        ),
        "Initialize": grpc.stream_unary_rpc_method_handler(
            servicer.Initialize,
            request_deserializer=pb_dot_competition__pb2.Oracle.FromString,
            response_serializer=pb_dot_competition__pb2.InitializationReply.SerializeToString,
        ),
        "Select": grpc.stream_stream_rpc_method_handler(
            servicer.Select,
            request_deserializer=pb_dot_competition__pb2.SDCTestCase.FromString,
            response_serializer=pb_dot_competition__pb2.SelectionReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "CompetitionTool", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers("CompetitionTool", rpc_method_handlers)


# This class is part of an EXPERIMENTAL API.
class CompetitionTool(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Name(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/CompetitionTool/Name",
            pb_dot_competition__pb2.Empty.SerializeToString,
            pb_dot_competition__pb2.NameReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def Initialize(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/CompetitionTool/Initialize",
            pb_dot_competition__pb2.Oracle.SerializeToString,
            pb_dot_competition__pb2.InitializationReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def Select(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/CompetitionTool/Select",
            pb_dot_competition__pb2.SDCTestCase.SerializeToString,
            pb_dot_competition__pb2.SelectionReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
