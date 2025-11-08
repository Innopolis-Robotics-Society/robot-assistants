#!/usr/bin/env python
# I am Zlobniy Lisia. I will kill all of you! oua-ha-ha-ha-ha!

from typing import List
import pygame as pg
import pygame._sdl2.controller

import URBasic
import URBasic.robotModel
import URBasic.urScriptExt

import json

SHIFT_COEF = 0.025
HOST = "172.31.1.26"

def main():

    robotModel = URBasic.robotModel.RobotModel()

    print("Initialization UR")

    robot = URBasic.urScriptExt.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.init_realtime_control()
    robot.set_realtime_pose([-0.675, 0.06,0.082,2.8,-1.026,0])
    print("Start position set")        
        
    workingZone = {
                    "minX": -1,
                    "maxX": -0.453,
                    "minY": -0.289,
                    "maxY": 0.326,
                    "minZ": -0.215,
                    "maxZ": 0.500,
                    }

        
    x_axis, y_axis, z_axis, actor_status = *robot.get_actual_tcp_pose_custom()[0:3], 0
    x_shift, y_shift, z_shift = 0,0,0
    pg.init()
    pygame._sdl2.controller.init()
    pygame._sdl2.controller.Controller(0).init()

    while True:
        for e in pg.event.get():
            if (e.type == 1536):
                
                joyValue = e.dict.get('value')*SHIFT_COEF
                
                if (e.dict.get('axis') == 0 ):
                    x_shift = -joyValue
                elif (e.dict.get('axis') == 1):
                    y_shift = joyValue
                elif (e.dict.get('axis') == 3):
                    z_shift = -joyValue
                    
            elif (e.type == 1539):
                
                if (e.dict.get('button') == 12):
                    actor_status = -1
                elif (e.dict.get('button') == 11):
                    actor_status = 1
                elif (e.type == 1540 and e.dict.get('button') in [11,12]):
                    actor_status = 0

        print(x_axis, y_axis, z_axis, actor_status, '[{: 06.4f}, {: 06.4f}, {: 06.4f},   {: 06.4f}, {: 06.4f}, {: 06.4f}]'.format(*robot.get_actual_tcp_pose_custom()))
        
        if ((workingZone["minX"] < robot.get_actual_tcp_pose_custom()[0] + x_shift and x_shift < 0) or (robot.get_actual_tcp_pose_custom()[0] + x_shift < workingZone["maxX"] and x_shift>=0)):
            x_axis = robot.get_actual_tcp_pose_custom()[0] + x_shift
        if ((workingZone["minY"] < robot.get_actual_tcp_pose_custom()[1] + y_shift and y_shift < 0) or (robot.get_actual_tcp_pose_custom()[1] + y_shift < workingZone["maxY"] and y_shift>=0)):
            y_axis = robot.get_actual_tcp_pose_custom()[1] + y_shift
        if ((workingZone["minZ"] < robot.get_actual_tcp_pose_custom()[2] + z_shift and z_shift < 0) or (robot.get_actual_tcp_pose_custom()[2] + z_shift < workingZone["maxZ"] and z_shift>=0)):
            z_axis = robot.get_actual_tcp_pose_custom()[2] + z_shift
        robot.set_realtime_pose([x_axis, y_axis,z_axis,2.8,-1.026,0])
                        
if __name__ == "__main__":
        main()
