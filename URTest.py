import URBasic
import URBasic.robotModel
import URBasic.urScriptExt
import time
HOST = "192.168.0.100"

robotModel = URBasic.robotModel.RobotModel()

print("Initialization UR")

robot = URBasic.urScriptExt.UrScriptExt(host=HOST, robotModel=robotModel)


robot.init_realtime_control()
print('Robot Pose: [{: 06.4f}, {: 06.4f}, {: 06.4f},   {: 06.4f}, {: 06.4f}, {: 06.4f}]'.format(*robot.get_actual_tcp_pose_custom()))
input('1?')
robot.set_realtime_pose([-0.643, 0.059,0.05,2.8,-1.026,0])
input()
robot.set_realtime_pose([-0.643, 0.059,0.1,2.8,-1.026,0])
input()
robot.set_realtime_pose([-0.643, 0.059,0.4,2.8,-1.026,2])
print("position send, close")
while 1:
    print('Robot Pose: [{: 06.4f}, {: 06.4f}, {: 06.4f},   {: 06.4f}, {: 06.4f}, {: 06.4f}]'.format(*robot.get_actual_tcp_pose_custom()))
    print(robot.get_actual_tcp_pose_custom())

robot.close()

"[-0.7860,  0.3171, -0.1065,    2.8000, -1.0260, -0.0002]"