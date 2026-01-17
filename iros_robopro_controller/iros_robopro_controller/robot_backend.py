from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass
class MotionParams:
    speed: float = 0.3
    accel: float = 1.0
    blend: float = 0.0
    orientation_units: str = 'deg'
    await_sec: int = 30


class RobotBackend:
    """Thin wrapper around RoboPro RobotApi.

    Notes:
      - This code expects vendor package available as: from API.rc_api import RobotApi
      - Poses are (x,y,z,rx,ry,rz) in meters & degrees (if orientation_units='deg').
    """

    def __init__(self, ip: str, timeout: int = 5, autoconnect: bool = False) -> None:
        from API.rc_api import RobotApi  # vendor

        self._ip = ip
        self._robot = RobotApi(ip, autoconnect=autoconnect, timeout=timeout, show_std_traceback=True)

    def connect(self, read_only: bool = False) -> bool:
        return bool(self._robot.connect(read_only=read_only))

    def is_connected(self) -> bool:
        return bool(self._robot.is_connected())

    def disconnect(self) -> bool:
        return bool(self._robot.disconnect())

    def set_tcp(self, tool_end_point: Sequence[float], units: Optional[str] = None) -> bool:
        return bool(self._robot.tool.set(tuple(tool_end_point), units))

    def get_tcp(self, units: Optional[str] = None) -> Tuple[float, float, float, float, float, float]:
        return tuple(self._robot.tool.get(units))  # type: ignore

    def get_actual_tcp(self, orientation_units: Optional[str] = None) -> Tuple[float, float, float, float, float, float]:
        return tuple(self._robot.motion.get_actual_position(orientation_units=orientation_units, position_format='tcp'))  # type: ignore

    def move_linear_tcp(self, tcp_pose: Sequence[float], mp: MotionParams) -> bool:
        if not self.is_connected():
            return False

        # Reset trajectory buffer
        ok = bool(self._robot.motion.mode.set('hold'))
        if not ok:
            return False

        ok = bool(
            self._robot.motion.linear.add_new_waypoint(
                tcp_pose=tuple(tcp_pose),
                speed=float(mp.speed),
                accel=float(mp.accel),
                blend=float(mp.blend),
                orientation_units=mp.orientation_units,
            )
        )
        if not ok:
            return False

        ok = bool(self._robot.motion.mode.set('move'))
        if not ok:
            return False

        return bool(self._robot.motion.wait_waypoint_completion(waypoint_count=0, await_sec=int(mp.await_sec)))

    def set_gripper_pins(self, do0: bool, do1: bool, backend: str = 'wrist', pin0: int = 0, pin1: int = 1) -> bool:
        """Set two digital outputs.

        backend:
          - 'wrist'      -> robot.wrist.digital.set_output(0..1)
          - 'controller' -> robot.io.digital.set_output(0..23)
        """
        if not self.is_connected():
            return False

        if backend == 'controller':
            ok0 = bool(self._robot.io.digital.set_output(int(pin0), bool(do0)))
            ok1 = bool(self._robot.io.digital.set_output(int(pin1), bool(do1)))
            return ok0 and ok1

        # default: wrist
        ok0 = bool(self._robot.wrist.digital.set_output(int(pin0), bool(do0)))
        ok1 = bool(self._robot.wrist.digital.set_output(int(pin1), bool(do1)))
        return ok0 and ok1

