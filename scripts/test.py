import time
import URBasic

HOST = "192.168.0.100"
rm = URBasic.robotModel.RobotModel()
ur = URBasic.urScriptExt.UrScriptExt(host=HOST, robotModel=rm)

try:
    print(1)
    # 0) Стоп на всякий случай (если что-то висело)
    try: ur.stopj(2.0)
    except: pass
    # опционально: ur.dashboardStop() если используете Dashboard
    print(2)
    # 1) Считать ТЕКУЩИЕ суставы/позу ДО realtime
    q_now = ur.get_actual_joint_positions()
    pose_now = ur.get_actual_tcp_pose()
    print(3)
    # 2) «Якорь» в текущие суставы малой скоростью
    ur.movej(q=q_now, a=0.5, v=0.1)   # фактически держит текущее положение
    time.sleep(0.05)
    print(4)
    # 3) Запустить realtime и МГНОВЕННО задать текущую цель
    ur.init_realtime_control()
    time.sleep(0.02)                  # дать петле стартануть
    print(5)
    ur.set_realtime_pose(pose_now)    # сразу фиксируем текущую позу
    time.sleep(0.05)
    print(6)

    # 5) Теперь можно двигать малыми шагами
    target = [-0.3583, -0.8985,  0.8708,    0.7062, -2.9221,  0.5281]
    ur.set_realtime_pose(target)
    print(8)
    time.sleep(0.5)

except Exception:
    try: ur.stopj(1.0)
    except: pass
    raise
finally:
    ur.close()
