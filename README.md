# robot-assistants (ROS 2 Humble)

Docker-first workspace for a lab Assistants project:

**UR10e + USB camera + AprilTags + YOLO + voice control + MoveIt (+ RViz)**

The main entrypoint launches a complete stack for the default demo scenario (pick/place behaviors).

---

## Supported environment

- **ROS 2:** Humble
- **Run mode:** Docker-first (CPU-only works; GPU optional for CV)
- **Robots:** UR10e (real robot or URSim)
- **Cameras:** USB UVC via `v4l2_camera` (depth cameras planned: RealSense / Azure Kinect)
- **Voice input:** host microphone (audio passthrough into container)
- **Gripper:** SCHUNK EGP-64 (controlled via ROS service; wired through a control box pins)

---

## Repository structure (packages)

- **`iros_assistant_bringup`**  
  Scenario-level launch files and “glue” nodes/services:
  - UR bringup wrapper (`launch/robots/ur.launch.py`)
  - demo scenario entrypoint (`launch/bringup.launch.py`)
  - tools (`launch/tools/*`), perception (`launch/perception/*`), world environment (`launch/world/*`)
  - C++ nodes: `go_to_frame`, `exec_ik_move`, `gripper_control`
  - services in `srv/*`

- **`iros_camera`**  
  Multi-camera launcher and camera registry:
  - `config/cameras.yaml` — camera list (device path, calibration file, TF, params)
  - `launch/camera.launch.py` — start one or multiple cameras by name
  - `calibration/*.yaml` — camera calibration files (`camera_info_url`)

- **`iros_april_tags`**  
  AprilTag detection + rectification pipeline:
  - `launch/detect_apriltags.launch.py` — namespaced pipeline per camera
  - `config/apriltag.yaml` — tag ids/frames/sizes and detector settings

- **`iros_tool_recognition`**  
  YOLO-based object localization node(s):
  - `ins_search.py` — detects an object, publishes debug image + target TF/Point
  - model stored in `models/`

- **`iros_ru_voice_recognision`**  
  Vosk-based Russian voice control:
  - `launch/voice_control.launch.py`
  - Vosk model stored in `models/`

- **`iros_assistant_behavior`**  
  Higher-level behavior node(s) (e.g., voice command executor).

---

## Quick start (Docker-first)

1. **Clone**
```bash
git clone https://github.com/Innopolis-Robotics-Society/robot-assistants.git
cd robot-assistants
```

2. **Build + run container**

* Use `docker-compose.yaml` / `Dockerfile` in the repo root.
* Make sure the container has access to:

  * camera devices (`/dev/video*` or `/dev/v4l/by-id/*`)
  * audio (`/dev/snd`)
  * (optional) GPU if available

Typical workflow:

```bash
docker compose up --terminal
```

Attach to the running container:

```bash
docker compose exec terminal bash
```

3. **Build workspace inside container**

```bash
colcon build --symlink-install
source install/setup.bash
```

---

## Configure cameras

Edit: **`iros_camera/config/cameras.yaml`**

Each camera is defined by a name and at minimum:

* `video_device`: a stable path recommended: `/dev/v4l/by-id/...`
* `camera_info`: calibration YAML path (relative to the `iros_camera` share dir)
* `frame_id`: child TF frame for this camera (recommended: `*_optical_frame`)
* `tf`: static transform (parent frame + xyz + rpy)
* `params`: optional per-camera driver params

Example shape:

```yaml
cameras:
  mook_laptop_camera:
    video_device: "/dev/v4l/by-id/usb-...-video-index0"
    camera_info: "calibration/mook_laptop_camera.yaml"
    frame_id: "mook_laptop_camera_optical_frame"
    tf:
      parent_frame: "base_link"
      xyz: [0.10, 0.05, 0.20]
      rpy: [0.0, 0.0, 0.0]
    params:
      image_size: [640, 480]
```

> If `image_proc/rectify_node` says the camera is **uncalibrated**, verify that:
>
> * `camera_info_url` points to the correct YAML
> * the driver publishes non-zero `CameraInfo` (`K/D/R/P`)
> * the camera resolution matches the calibration (`image_width/height`)

---

## Main demo (one command)

**Entrypoint:**

```bash
ros2 launch iros_assistant_bringup bringup.launch.py robot_ip:=<ROBOT_OR_URSIM_IP>
```

Common arguments you’ll typically use:

* `robot_ip:=...` (UR10e)
* `ur_type:=ur10e`
* camera selection (passed into the camera + perception parts):

  * `cameras:=mook_laptop_camera` (single)
  * `cameras:=mook_laptop_camera,marat_camera` (multiple)
* multi-robot namespaces (future): `namespace:=robot1` etc.

> The assistant scenario currently assumes **all components are enabled**:
> camera, rectification, AprilTags, YOLO, voice, MoveIt, RViz.

---

## Running individual subsystems

### Camera only

```bash
ros2 launch iros_camera camera.launch.py cameras:=mook_laptop_camera
```

### AprilTags (for a specific camera namespace)

/home/mobile/ros2_ws/src/iros_cv_algorithms/iros_cv_algorithms/algos/models/yolo11s_best.pt

```bash
ros2 launch iros_april_tags detect_apriltags.launch.py camera:=mook_laptop_camera
```

### CV node (YOLO)

If your CV launch expects rectified images:

* ensure `image_rect` and `camera_info` are available (often via the AprilTags pipeline)

---

## Services (iros_assistant_bringup)

Service definitions live in: `iros_custom_msgs/srv/`

### `DetectObject` (`iros_custom_msgs/srv/DetectObject.srv`)

Request:
```text
string class_name   # class name from model.names (e.g. "hammer")
float32 duration    # tracking duration in seconds; <= 0 means "until stopped"
````

Response:

```text
bool accepted
string message
```

Typical usage:

* Starts/arms the YOLO-based detector to track a specific object class for a limited time window or indefinitely.
* The node usually publishes:

  * debug image topic (overlay / mask)
  * `PointStamped` with the measured target point
  * a TF frame for the detected target (if enabled in the node)

Example call:

```bash
ros2 service call /detect_object iros_custom_msgs/srv/DetectObject "{class_name: 'hammer', duration: 5.0}"
```

---

### `GoToFrame` (`iros_custom_msgs/srv/GoToFrame.srv`)

Request:

```text
string frame
```

Response:

```text
bool success
string message
```

Typical usage:

* Commands the robot to move to a target defined by a TF frame (e.g. `pose_forward`, `pose_watering`, or a perception-generated frame like `hoba_target`).

Example call:

```bash
ros2 service call /go_to_frame iros_custom_msgs/srv/GoToFrame "{frame: 'pose_forward'}"
```

---

### `GripperAction` (`iros_custom_msgs/srv/GripperAction.srv`)

Request:

```text
bool open   # true = open, false = close
```

Response:

```text
bool success
string message
```

Typical usage:

* Opens/closes the SCHUNK EGP-64 gripper via control box wiring.

Examples:

```bash
ros2 service call /gripper_action iros_custom_msgs/srv/GripperAction "{open: true}"
ros2 service call /gripper_action iros_custom_msgs/srv/GripperAction "{open: false}"
```

---

### `MoveToPose` (`iros_custom_msgs/srv/MoveToPose.srv`)

Request:

```text
geometry_msgs/Pose target
```

Response:

```text
bool success
string message
```

Typical usage:

* Commands the robot to move to an explicit Cartesian pose target (as a `geometry_msgs/Pose`), typically using MoveIt.

Example call:

```bash
ros2 service call /move_to_pose iros_custom_msgs/srv/MoveToPose "{
  target: {
    position: {x: 0.4, y: 0.0, z: 0.3},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}"
```


---

## Notes on TF / frames (important)

* Cameras are launched in their **own namespace** (camera name). Topics look like:

  * `/<camera>/image_raw`
  * `/<camera>/camera_info`
  * `/<camera>/image_rect` (if rectification is running)

* AprilTag TF frame naming can depend on:

  * tag family (`tag36h11`, etc.)
  * your `apriltag.yaml` mapping (`tag.ids`, `tag.frames`, `tag.sizes`)
  * `apriltag_ros` parameters controlling TF publishing

If detections exist but TF is missing, verify in the AprilTag node parameters that TF publishing is enabled and that `camera_info` is calibrated (non-zero intrinsics).

---

## Scripts

* `scripts/set_ur_connection.sh` — helper for UR connection setup (host-side)
* `scripts/install_vosk_model.sh` — helper for Vosk model installation (if needed outside Docker)

---

## Status

Internal lab project with a minimal public-facing structure. APIs and launch arguments may evolve as multi-robot scenarios are added.
