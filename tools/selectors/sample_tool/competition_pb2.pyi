from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

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
    __slots__ = ("testCase", "hasFailed", "executionTime", "failureDetails", "metrics")
    TESTCASE_FIELD_NUMBER: _ClassVar[int]
    HASFAILED_FIELD_NUMBER: _ClassVar[int]
    EXECUTIONTIME_FIELD_NUMBER: _ClassVar[int]
    FAILUREDETAILS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    testCase: SDCTestCase
    hasFailed: bool
    executionTime: float
    failureDetails: str
    metrics: TestMetrics
    def __init__(self, testCase: _Optional[_Union[SDCTestCase, _Mapping]] = ..., hasFailed: bool = ..., executionTime: _Optional[float] = ..., failureDetails: _Optional[str] = ..., metrics: _Optional[_Union[TestMetrics, _Mapping]] = ...) -> None: ...

class TestMetrics(_message.Message):
    __slots__ = ("realityGapScore", "visualFidelity", "computationalCost", "testingEffectiveness", "safetyRisk", "humanAlignment", "oracleReliability", "automationLevel")
    REALITYGAPSCORE_FIELD_NUMBER: _ClassVar[int]
    VISUALFIDELITY_FIELD_NUMBER: _ClassVar[int]
    COMPUTATIONALCOST_FIELD_NUMBER: _ClassVar[int]
    TESTINGEFFECTIVENESS_FIELD_NUMBER: _ClassVar[int]
    SAFETYRISK_FIELD_NUMBER: _ClassVar[int]
    HUMANALIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ORACLERELIABILITY_FIELD_NUMBER: _ClassVar[int]
    AUTOMATIONLEVEL_FIELD_NUMBER: _ClassVar[int]
    realityGapScore: float
    visualFidelity: float
    computationalCost: float
    testingEffectiveness: float
    safetyRisk: float
    humanAlignment: float
    oracleReliability: float
    automationLevel: float
    def __init__(self, realityGapScore: _Optional[float] = ..., visualFidelity: _Optional[float] = ..., computationalCost: _Optional[float] = ..., testingEffectiveness: _Optional[float] = ..., safetyRisk: _Optional[float] = ..., humanAlignment: _Optional[float] = ..., oracleReliability: _Optional[float] = ..., automationLevel: _Optional[float] = ...) -> None: ...

class SDCTestCase(_message.Message):
    __slots__ = ("testId", "roadPoints")
    TESTID_FIELD_NUMBER: _ClassVar[int]
    ROADPOINTS_FIELD_NUMBER: _ClassVar[int]
    testId: str
    roadPoints: _containers.RepeatedCompositeFieldContainer[RoadPoint]
    def __init__(self, testId: _Optional[str] = ..., roadPoints: _Optional[_Iterable[_Union[RoadPoint, _Mapping]]] = ...) -> None: ...

class RoadPoint(_message.Message):
    __slots__ = ("sequenceNumber", "x", "y")
    SEQUENCENUMBER_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    sequenceNumber: int
    x: float
    y: float
    def __init__(self, sequenceNumber: _Optional[int] = ..., x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class InitializationReply(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool = ..., message: _Optional[str] = ...) -> None: ...

class SelectionReply(_message.Message):
    __slots__ = ("testId", "confidence", "reasoning")
    TESTID_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    testId: str
    confidence: float
    reasoning: str
    def __init__(self, testId: _Optional[str] = ..., confidence: _Optional[float] = ..., reasoning: _Optional[str] = ...) -> None: ...

class MetricsSupport(_message.Message):
    __slots__ = ("supportsRealityGap", "supportsCostEffectiveness", "supportsSafetyAssessment", "supportsOracleQuality", "toolVersion")
    SUPPORTSREALITYGAP_FIELD_NUMBER: _ClassVar[int]
    SUPPORTSCOSTEFFECTIVENESS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTSSAFETYASSESSMENT_FIELD_NUMBER: _ClassVar[int]
    SUPPORTSORACLEQUALITY_FIELD_NUMBER: _ClassVar[int]
    TOOLVERSION_FIELD_NUMBER: _ClassVar[int]
    supportsRealityGap: bool
    supportsCostEffectiveness: bool
    supportsSafetyAssessment: bool
    supportsOracleQuality: bool
    toolVersion: str
    def __init__(self, supportsRealityGap: bool = ..., supportsCostEffectiveness: bool = ..., supportsSafetyAssessment: bool = ..., supportsOracleQuality: bool = ..., toolVersion: _Optional[str] = ...) -> None: ...
