import gymnasium as gym
import mani_skill.envs
from lerobot.motors.feetech import FeetechMotorsBus
import numpy as np
import time
from types import SimpleNamespace

env = gym.make('StackCube-v1', 
               robot_uids="panda", 
               render_mode="human", 
               control_mode="pd_joint_pos")
obs, _ = env.reset()
print("Environment created successfully")

port = '/dev/ttyACM0'
motor_definitions = {
        'shoulder_pan': SimpleNamespace(id=1, model='sts3215'),
        'shoulder_lift': SimpleNamespace(id=2, model='sts3215'),
        'elbow_flex': SimpleNamespace(id=3, model='sts3215'),
        'wrist_flex': SimpleNamespace(id=4, model='sts3215'),
        'wrist_roll': SimpleNamespace(id=5, model='sts3215'),
        'gripper': SimpleNamespace(id=6, model='sts3215')
    }

leader = FeetechMotorsBus(port, motor_definitions)
leader.connect()
print("Physical leader connected successfully")


try:
    while True:
        pos_dict = {}
        for motor_name in motor_definitions.keys():
            pos_dict[motor_name] = leader.read(data_name="Present_Position",
                                               motor=motor_name,
                                               normalize=False)

        q_raw = np.array([pos_dict[motor_name] for motor_name in motor_definitions.keys()])    
        action_6 = (q_raw / 2048.0) - 1.0
        action_6 = np.clip(action_6, -1.0, 1.0)

        action_8 = np.zeros(8)
        action_8[:6] = action_6
        obs, reward, terminated, truncated, info = env.step(action_8)

        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            print("Episode ended, resetting environment")
except KeyboardInterrupt:
    print("Leader disconnected successfully")
    leader.disconnect()