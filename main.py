import torch
import argparse
from  common.poselive import PoseLive

if __name__ == "__main__":
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 27
    args.number_of_kept_frames, args.number_of_kept_coeffs = 1, 3
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'model\checkpoint'
    args.n_joints, args.out_joints = 17, 17

    model = PoseLive()
    model.load(args)
    model.run_inference()

