import importlib
hjcdik = importlib.import_module("hjcdik")
import numpy as np
import zmq
import sys
import os
import scipy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def main():
    # setup hjcdik
    N = hjcdik.num_joints()
    print(f"[info] robot with {N} joints")
    T, batches, S = 100, 1000, 1

    # warmup
    warmup_target = hjcdik.sample_targets(T, seed=0 + 12345)[0]
    hjcdik.generate_solutions(warmup_target, batch_size=batches, num_solutions=1)

    # initialize socket
    context = zmq.Context()
    clientsocket = context.socket(zmq.REQ)
    clientsocket.connect("tcp://172.16.0.1:6769")
    send_array(clientsocket, np.zeros(1))

    input("Start tracking >> ")
    last_pose = None

    apriltag_transform = np.array([[1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, -1]])

    while True:
        try:
            pose = recv_array(clientsocket)
            if last_pose is None:
                last_pose = pose

            # do some transformation to R before conversion
            # apriltag z points out, rotate around x by 180
            quat = []
            if np.count_nonzero(pose) != 0:
                R = pose[0:3, 0:3]
                R = scipy.spatial.transform.Rotation.from_matrix(R)
                quat = R.as_quat(scalar_first = True)
            else:
                quat = np.array([0.000000, 1, 0.000000, 0])

            pose_arr = np.concatenate([pose[0:3, 3], quat])
            # add pose offset
            pose_arr[2] += 0.3
            # pose_quat = np.array([pose[0, 3], pose[1, 3], pose[2, 3] + 0.3, 0.000000, 1, 0.000000, 0])
            res = hjcdik.generate_solutions(pose_arr, batch_size=batches, num_solutions=1000)
            Q = res["joint_config"]
            PErr = res["pos_errors"]
            OErr = res["ori_errors"]
            send_array(clientsocket, Q)
            recv_array(clientsocket)
            send_array(clientsocket, PErr)
            recv_array(clientsocket)
            send_array(clientsocket, OErr)
        except Exception as e:
            print(f"Error: {e}")
            exit()

            


if __name__ == "__main__":
    main()
