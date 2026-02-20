import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external.AnDRLib import andrlib
import numpy as np
import zmq
import ruckig

default_configuration = np.array([-0.008553700521588326, -1.0634461641311646, -0.0247645266354084, -2.9395430088043213, -0.01800784468650818, 1.9824919700622559, 0.7801224589347839])

def to_robot_frame(pose, cam_to_robot):
    try:
        return cam_to_robot @ pose
    except:
        return np.zeros((4, 4))

def send_array(
    socket: zmq.Socket,
    A: np.ndarray,
    flags: int = 0,
    **kwargs,
):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, **kwargs)

def recv_array(socket: zmq.Socket, flags: int = 0, **kwargs) -> np.ndarray:
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, **kwargs)
    A = np.frombuffer(msg, dtype=md["dtype"])
    return A.reshape(md["shape"])

def control_loop():
    # Listen for incoming connection
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:6769")
    print("Waiting for a connection...")
    ack = recv_array(socket)
    print(ack)
    print(f"Connection established")

    # on successful connection, start control
    panda = andrlib.Panda("172.16.0.2")
    rs0 = andrlib.Realsense("234322070337")
    rs1 = andrlib.Realsense("234322070341")
    cam_to_robot0 = rs0.get_camera_to_robot_transform()
    while np.count_nonzero(cam_to_robot0) == 0:
        print("Waiting for camera to robot transform...")
        cam_to_robot0 = rs0.get_camera_to_robot_transform()

    cam_to_robot1 = rs1.get_camera_to_robot_transform()
    while np.count_nonzero(cam_to_robot1) == 0:
        print("Waiting for camera to robot transform...")
        cam_to_robot1 = rs1.get_camera_to_robot_transform()

    print("Camera to robot transform:")
    print(cam_to_robot1)
    
    inp = ruckig.InputParameter(7)
    inp.max_velocity = np.array([1] * 7)
    inp.max_acceleration = np.array([20.0] * 7)
    inp.max_jerk = np.array([40.0] * 7)
    panda.start_control(1)
    last_pose = None
    last_q = None
    pose = None

    while True:
        pose0 = to_robot_frame(rs0.get_apriltag_pose(1, 0.075), cam_to_robot0)
        pose1 = to_robot_frame(rs1.get_apriltag_pose(1, 0.075), cam_to_robot1)
        if np.count_nonzero(pose0) != 0:
            send_array(socket, pose0)
            pose = pose0
        else:
            send_array(socket, pose1)
            pose = pose1
        if last_pose is None:
            last_pose = pose
        # receive IK poses
        Q = recv_array(socket)
        send_array(socket, np.zeros(1))
        PErr = recv_array(socket)
        send_array(socket, np.zeros(1))
        OErr = recv_array(socket)

        q = np.array(panda.get_pose())

        diffs = Q - q
        dists = np.linalg.norm(diffs, axis=1)
        cands = np.where((PErr < 1e-5) & (OErr < 1e-1))[0]
        Q = Q[cands] if len(cands) > 0 else Q
        idx = np.argmin(dists[cands]) if len(cands) > 0 else np.argmin(dists)
        if last_q is None or \
            (np.linalg.norm(pose - last_pose) > 0.02 and \
                np.count_nonzero(pose) > 0):
            last_q = Q[idx]
            last_pose = pose

        panda.pause()
        inp.current_position = np.array(panda.get_pose())
        inp.current_velocity = np.array(panda.get_vel())
        inp.current_acceleration = np.array(panda.get_accel())

        inp.target_position = last_q
        inp.target_velocity = np.zeros(7)
        inp.target_acceleration = np.zeros(7)

        try:
            otg = ruckig.Ruckig(7)
            trajectory = ruckig.Trajectory(7)
            result = otg.calculate(inp, trajectory)
            q_traj = []
            qd_traj = []
            qdd_traj = []
            for i in range(int(trajectory.duration * 1000)):
                q_traj.append(trajectory.at_time(i / 1000.0)[0])
                qd_traj.append(trajectory.at_time(i / 1000.0)[1])
                qdd_traj.append(trajectory.at_time(i / 1000.0)[2])
            q_traj = np.array(q_traj)[0:200]
            qd_traj = np.array(qd_traj)[0:200]
            qdd_traj = np.array(qdd_traj)[0:200]
            panda.set_all(q_traj, qd_traj, qdd_traj)
            panda.unpause()
            time.sleep(0.01)
        except Exception as e:
            print("Error: ", e)

if __name__ == "__main__":
    control_loop()