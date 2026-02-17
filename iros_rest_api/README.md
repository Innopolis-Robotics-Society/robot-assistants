# iros_rest_api

FastAPI gateway for the existing `robot-assistants` ROS2 stack.

It does **not** replace the current ROS2 architecture. It adds an HTTP layer that can be called from curl, browser Swagger UI, frontend apps, external orchestration scripts, or demos.

## What it exposes

| HTTP endpoint | ROS2 target |
|---|---|
| `GET /health` | checks known ROS services |
| `GET /ros/services` | lists ROS services visible to the node |
| `GET /state` | latest subscribed state from `/cv_hub/*` and `voice/executor_status` |
| `POST /perception/detect-object` | calls `/detect_object` (`iros_custom_msgs/srv/DetectObject`) |
| `POST /robot/go-to-frame` | calls `/go_to_frame` (`iros_custom_msgs/srv/GoToFrame`) |
| `POST /robot/gripper` | calls `/gripper_action` (`iros_custom_msgs/srv/GripperAction`) |
| `POST /voice/command` | publishes `std_msgs/String` to `voice/command` |
| `POST /cv/run` | calls CV hub services: rust / pcb / gear / publish |
| `GET /commands/{command_id}` | tracks async REST commands |

## Install Python dependencies

Inside the ROS2 container:

```bash
python3 -m pip install -r src/iros_rest_api/requirements.txt
```

## Build

From `/home/mobile/ros2_ws`:

```bash
colcon build --symlink-install --packages-select iros_custom_msgs iros_rest_api
source install/setup.bash
```

## Run

```bash
ros2 run iros_rest_api api_server --host 0.0.0.0 --port 8000
```

or:

```bash
ros2 launch iros_rest_api rest_api.launch.py host:=0.0.0.0 port:=8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Example calls

Health:

```bash
curl http://localhost:8000/health
```

Run tool detection:

```bash
curl -X POST http://localhost:8000/perception/detect-object \
  -H 'Content-Type: application/json' \
  -d '{"class_name":"hammer","duration":5.0,"timeout_s":10.0}'
```

Move robot to a TF frame:

```bash
curl -X POST http://localhost:8000/robot/go-to-frame \
  -H 'Content-Type: application/json' \
  -d '{"frame":"pose_forward","timeout_s":30.0}'
```

Open gripper:

```bash
curl -X POST http://localhost:8000/robot/gripper \
  -H 'Content-Type: application/json' \
  -d '{"open":true}'
```

Send existing voice command through HTTP:

```bash
curl -X POST http://localhost:8000/voice/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"молоток"}'
```

Run CV hub gear check asynchronously:

```bash
curl -X POST http://localhost:8000/cv/run \
  -H 'Content-Type: application/json' \
  -d '{"target":"gear","timeout_s":20.0,"async_execution":true}'
```

Then check status:

```bash
curl http://localhost:8000/commands/<command_id>
```
