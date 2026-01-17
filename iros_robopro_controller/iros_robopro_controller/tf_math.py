from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Transform:
    """Rigid transform represented as translation + quaternion.

    Quaternion is (x, y, z, w).
    """

    t: np.ndarray  # (3,)
    q: np.ndarray  # (4,)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=float)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_normalize(np.array([x, y, z, w], dtype=float))


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (rad), applied in XYZ order -> quaternion (x,y,z,w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # intrinsic rotations about X,Y,Z
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([x, y, z, w], dtype=float))


def quat_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    """Quaternion (x,y,z,w) -> roll,pitch,yaw in radians (XYZ / RPY)."""
    x, y, z, w = quat_normalize(q)

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(_clamp(sinp, -1.0, 1.0))

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def transform_to_matrix(T: Transform) -> np.ndarray:
    R = quat_to_rot(T.q)
    M = np.eye(4, dtype=float)
    M[0:3, 0:3] = R
    M[0:3, 3] = np.asarray(T.t, dtype=float)
    return M


def matrix_to_transform(M: np.ndarray) -> Transform:
    M = np.asarray(M, dtype=float)
    t = M[0:3, 3].copy()
    q = rot_to_quat(M[0:3, 0:3])
    return Transform(t=t, q=q)


def invert(T: Transform) -> Transform:
    M = transform_to_matrix(T)
    R = M[0:3, 0:3]
    t = M[0:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    q_inv = quat_normalize(np.array([-T.q[0], -T.q[1], -T.q[2], T.q[3]], dtype=float))
    return Transform(t=t_inv, q=q_inv)


def multiply(A: Transform, B: Transform) -> Transform:
    """Compose: A * B"""
    MA = transform_to_matrix(A)
    MB = transform_to_matrix(B)
    return matrix_to_transform(MA @ MB)


def robot_pose_deg_to_transform(pose: Tuple[float, float, float, float, float, float]) -> Transform:
    """(x,y,z,rx,ry,rz) with angles in degrees -> Transform."""
    x, y, z, rx, ry, rz = pose
    q = rpy_to_quat(math.radians(rx), math.radians(ry), math.radians(rz))
    return Transform(t=np.array([x, y, z], dtype=float), q=q)


def transform_to_robot_pose_deg(T: Transform) -> Tuple[float, float, float, float, float, float]:
    """Transform -> (x,y,z,rx,ry,rz) in degrees."""
    roll, pitch, yaw = quat_to_rpy(T.q)
    return (
        float(T.t[0]),
        float(T.t[1]),
        float(T.t[2]),
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )
