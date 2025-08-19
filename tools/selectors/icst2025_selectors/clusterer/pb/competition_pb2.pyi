from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NameReply(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class Oracle(_message.Message):
    __slots__ = ("testCase", "hasFailed")
    TESTCASE_FIELD_NUMBER: _ClassVar[int]
    HASFAILED_FIELD_NUMBER: _ClassVar[int]
    testCase: SDCTestCase
    hasFailed: bool
    def __init__(
        self,
        testCase: _Optional[_Union[SDCTestCase, _Mapping]] = ...,
        hasFailed: bool = ...,
    ) -> None: ...

class SDCTestCase(_message.Message):
    __slots__ = ("testId", "roadPoints")
    TESTID_FIELD_NUMBER: _ClassVar[int]
    ROADPOINTS_FIELD_NUMBER: _ClassVar[int]
    testId: str
    roadPoints: _containers.RepeatedCompositeFieldContainer[RoadPoint]
    def __init__(
        self,
        testId: _Optional[str] = ...,
        roadPoints: _Optional[_Iterable[_Union[RoadPoint, _Mapping]]] = ...,
    ) -> None: ...

class RoadPoint(_message.Message):
    __slots__ = ("sequenceNumber", "x", "y")
    SEQUENCENUMBER_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    sequenceNumber: int
    x: float
    y: float
    def __init__(
        self,
        sequenceNumber: _Optional[int] = ...,
        x: _Optional[float] = ...,
        y: _Optional[float] = ...,
    ) -> None: ...

class InitializationReply(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool = ...) -> None: ...

class SelectionReply(_message.Message):
    __slots__ = ("testId",)
    TESTID_FIELD_NUMBER: _ClassVar[int]
    testId: str
    def __init__(self, testId: _Optional[str] = ...) -> None: ...
