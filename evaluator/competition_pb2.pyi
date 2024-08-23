from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class NameReply(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class SDCTestCase(_message.Message):
    __slots__ = ("testId",)
    TESTID_FIELD_NUMBER: _ClassVar[int]
    testId: str
    def __init__(self, testId: _Optional[str] = ...) -> None: ...

class InitializationReply(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SelectionReply(_message.Message):
    __slots__ = ("testId",)
    TESTID_FIELD_NUMBER: _ClassVar[int]
    testId: str
    def __init__(self, testId: _Optional[str] = ...) -> None: ...
