import torch
from omegaconf import DictConfig, OmegaConf
import hydra, os
import numpy as np
import gc
import copy
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import glob

def load_file_path(task_path):
    task_files = glob.glob(os.path.join(task_path, "traj*.pkl"))
    return sorted(task_files)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str)
    args = parser.parse_args()

    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    files_path = load_file_path(task_path=args.task_path)

    for file in files_path:
        with open(file, 'rb') as f:
            traj_data = pickle.load(f)
            traj = traj_data['traj']
            for t in range(len(traj)):
                traj_time_stamp = traj.get(t)
                if t>0:
                    print(traj_time_stamp['action'])

