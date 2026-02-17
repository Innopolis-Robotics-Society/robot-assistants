from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ApiStatus(BaseModel):
    status: Literal["ok", "degraded"]
    uptime_s: float
    ros_ready: bool
    node_name: str
    known_services: Dict[str, bool]
    last_error: Optional[str] = None


class ServiceCallResult(BaseModel):
    success: bool
    message: str = ""
    service_name: str
    latency_ms: float
    raw: Dict[str, Any] = Field(default_factory=dict)


class AsyncAccepted(BaseModel):
    command_id: str
    status: str
    kind: str


class CommandRecord(BaseModel):
    command_id: str
    kind: str
    status: Literal["pending", "running", "success", "failed"]
    created_at: float
    updated_at: float
    request: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DetectObjectRequest(BaseModel):
    class_name: str = Field(..., min_length=1, description="Class name from YOLO model.names, for example 'hammer'.")
    duration: float = Field(5.0, ge=0.0, description="Tracking duration in seconds; 0 means detector-specific default/infinite mode.")
    timeout_s: float = Field(10.0, gt=0.0, le=120.0)
    async_execution: bool = Field(False, description="If true, return command_id immediately and run the ROS service call in background.")


class GoToFrameRequest(BaseModel):
    frame: str = Field(..., min_length=1, description="Target TF frame, for example 'pose_forward' or 'hoba_target'.")
    timeout_s: float = Field(30.0, gt=0.0, le=300.0)
    async_execution: bool = False


class GripperRequest(BaseModel):
    open: bool = Field(..., description="true = open gripper, false = close gripper")
    timeout_s: float = Field(10.0, gt=0.0, le=120.0)
    async_execution: bool = False


class VoiceCommandRequest(BaseModel):
    command: str = Field(..., min_length=1, description="Text command published to voice/command topic.")
    topic: str = Field("voice/command", min_length=1)


class CvRunRequest(BaseModel):
    target: Literal["rust", "pcb", "gear", "publish"]
    timeout_s: float = Field(20.0, gt=0.0, le=300.0)
    async_execution: bool = False


class RosTopicSnapshot(BaseModel):
    cv_ok: Optional[bool] = None
    cv_report: Optional[str] = None
    voice_executor_status: Optional[str] = None
    updated_at: Dict[str, float] = Field(default_factory=dict)


class ServiceInfo(BaseModel):
    name: str
    types: List[str]
