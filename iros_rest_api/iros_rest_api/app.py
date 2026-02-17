from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, HTTPException

from .ros_bridge import RosApiBridge
from .schemas import (
    ApiStatus,
    AsyncAccepted,
    CommandRecord,
    CvRunRequest,
    DetectObjectRequest,
    GoToFrameRequest,
    GripperRequest,
    RosTopicSnapshot,
    ServiceCallResult,
    ServiceInfo,
    VoiceCommandRequest,
)


def create_app(bridge: RosApiBridge | None = None) -> FastAPI:
    bridge = bridge or RosApiBridge()

    app = FastAPI(
        title="Robot Assistants REST API",
        version="0.1.0",
        description=(
            "REST gateway over the existing ROS2 robot-assistants stack. "
            "It exposes robot movement, gripper, object detection, CV session hub "
            "and voice-command operations as HTTP endpoints."
        ),
    )
    app.state.bridge = bridge

    @app.on_event("startup")
    def _startup() -> None:
        app.state.bridge.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.bridge.stop()

    @app.get("/", include_in_schema=False)
    def root() -> Dict[str, str]:
        return {
            "name": "Robot Assistants REST API",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=ApiStatus, tags=["system"])
    def health() -> ApiStatus:
        known = app.state.bridge.known_service_availability()
        degraded = app.state.bridge.last_error is not None
        return ApiStatus(
            status="degraded" if degraded else "ok",
            uptime_s=app.state.bridge.uptime_s,
            ros_ready=True,
            node_name=app.state.bridge.node_name,
            known_services=known,
            last_error=app.state.bridge.last_error,
        )

    @app.get("/ros/services", response_model=List[ServiceInfo], tags=["system"])
    def list_services() -> List[ServiceInfo]:
        return [ServiceInfo(name=name, types=types) for name, types in app.state.bridge.list_services()]

    @app.get("/state", response_model=RosTopicSnapshot, tags=["system"])
    def state() -> RosTopicSnapshot:
        return RosTopicSnapshot(**app.state.bridge.snapshot())

    @app.get("/commands", response_model=List[CommandRecord], tags=["commands"])
    def list_commands() -> List[CommandRecord]:
        return app.state.bridge.list_commands()

    @app.get("/commands/{command_id}", response_model=CommandRecord, tags=["commands"])
    def get_command(command_id: str) -> CommandRecord:
        record = app.state.bridge.get_command(command_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown command_id: {command_id}")
        return record

    @app.post("/perception/detect-object", response_model=ServiceCallResult | AsyncAccepted, tags=["perception"])
    def detect_object(req: DetectObjectRequest):
        payload = req.dict()
        if req.async_execution:
            record = app.state.bridge.submit_command(
                kind="detect_object",
                request=payload,
                fn=lambda: app.state.bridge.detect_object(req.class_name, req.duration, req.timeout_s),
            )
            return AsyncAccepted(command_id=record.command_id, status=record.status, kind=record.kind)
        return app.state.bridge.detect_object(req.class_name, req.duration, req.timeout_s)

    @app.post("/robot/go-to-frame", response_model=ServiceCallResult | AsyncAccepted, tags=["robot"])
    def go_to_frame(req: GoToFrameRequest):
        payload = req.dict()
        if req.async_execution:
            record = app.state.bridge.submit_command(
                kind="go_to_frame",
                request=payload,
                fn=lambda: app.state.bridge.go_to_frame(req.frame, req.timeout_s),
            )
            return AsyncAccepted(command_id=record.command_id, status=record.status, kind=record.kind)
        return app.state.bridge.go_to_frame(req.frame, req.timeout_s)

    @app.post("/robot/gripper", response_model=ServiceCallResult | AsyncAccepted, tags=["robot"])
    def gripper(req: GripperRequest):
        payload = req.dict()
        if req.async_execution:
            record = app.state.bridge.submit_command(
                kind="gripper",
                request=payload,
                fn=lambda: app.state.bridge.gripper(req.open, req.timeout_s),
            )
            return AsyncAccepted(command_id=record.command_id, status=record.status, kind=record.kind)
        return app.state.bridge.gripper(req.open, req.timeout_s)

    @app.post("/voice/command", tags=["voice"])
    def voice_command(req: VoiceCommandRequest) -> Dict[str, object]:
        return app.state.bridge.publish_voice_command(req.command, req.topic)

    @app.post("/cv/run", response_model=ServiceCallResult | AsyncAccepted, tags=["cv"])
    def run_cv(req: CvRunRequest):
        payload = req.dict()
        if req.async_execution:
            record = app.state.bridge.submit_command(
                kind=f"cv_{req.target}",
                request=payload,
                fn=lambda: app.state.bridge.run_cv(req.target, req.timeout_s),
            )
            return AsyncAccepted(command_id=record.command_id, status=record.status, kind=record.kind)
        return app.state.bridge.run_cv(req.target, req.timeout_s)

    @app.get("/cv/report", response_model=RosTopicSnapshot, tags=["cv"])
    def cv_report() -> RosTopicSnapshot:
        return RosTopicSnapshot(**app.state.bridge.snapshot())

    return app


app = create_app()
