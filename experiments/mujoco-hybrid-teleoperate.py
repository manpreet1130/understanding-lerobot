import mujoco
import mujoco.viewer    
import numpy as np
import time
from lerobot.motors.feetech import FeetechMotorsBus
from types import SimpleNamespace


model = mujoco.MjModel.from_xml_path("assets/mujoco/so100.xml")
data = mujoco.MjData(model)

port = "/dev/ttyACM0"

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

joint_order = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper'
]

def normalize(q):
    return (q / 2048.0) - 1.0

print("Starting SO100 -> Mujoco teleop...")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        prev_q = np.zeros(6)
        alpha = 0.2  # smoothing

        while viewer.is_running():

            # ----------------------------
            # READ REAL ROBOT
            # ----------------------------
            pos_dict = {}

            for m in motor_definitions:
                pos_dict[m] = leader.read(
                    data_name="Present_Position",
                    motor=m,
                    normalize=False
                )

            # ----------------------------
            # BUILD JOINT VECTOR
            # ----------------------------
            q = np.array([pos_dict[j] for j in joint_order])

            # normalize (temporary)
            q = normalize(q)

            # smoothing (VERY important)
            q = alpha * q + (1 - alpha) * prev_q
            prev_q = q.copy()

            # ----------------------------
            # WRITE TO MUJOCO
            # ----------------------------
            data.qpos[:] = q

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping teleop...")

finally:
    leader.disconnect()
    print("Leader disconnected")
