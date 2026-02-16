import importlib
import ruckig
hjcdik = importlib.import_module("hjcdik")
# from external.AnDRLib.build import andrlib

def main():
    # setup hjcdik
    N = hjcdik.num_joints()
    print(f"[info] robot with {N} joints")
    T, batches, S = 100, 1000, 1
    targets = hjcdik.sample_targets(T, seed=0)

    # warmup
    warmup_target = hjcdik.sample_targets(1, seed=0 + 12345)[0]
    _ = hjcdik.generate_solutions(warmup_target, batch_size=batches, num_solutions=1)

    y_batch, y_time_ms, y_pos, y_ori = [], [], [], []

    # get ik solution
    res = hjcdik.generate_solutions(targets[0], batch_size=batches, num_solutions=100)
    print(res)

if __name__ == "__main__":
    main()
