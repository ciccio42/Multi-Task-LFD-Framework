import pickle
import os
import cv2
import debugpy
import logging
from PIL import Image
import numpy as np
import random
import glob

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

MAX_DIST = 10
MIN_DIST = 0.01

# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
# object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

ranges = np.array([[float('inf'), float('-inf')],
                   [float('inf'), float('-inf')],
                   [float('inf'), float('-inf')],
                   [float('inf'), float('-inf')],
                   [float('inf'), float('-inf')],
                   [float('inf'), float('-inf')]])
# p_x_min = float('inf')
# p_x_max = float('-inf')
# p_y_min = float('inf')
# p_y_max = float('-inf')
# p_z_min = float('inf')
# p_z_max = float('-inf')
# p_e_x_min = float('inf')
# p_e_x_max = float('-inf')
# p_e_y_min = float('inf')
# p_e_y_max = float('-inf')
# p_e_z_min = float('inf')
# p_e_z_max = float('-inf')

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument("--debug", action='store_true',
                        help="whether or not attach the debugger")
    parser.add_argument("--depth", action='store_true',
                        help="whether or not render depth")
    args = parser.parse_args()

    camera_name = "camera_front"

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()

    logger.info(f"Task path: {args.task_path}")
    i = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    task_paths = glob.glob(os.path.join(args.task_path, 'task_*'))
    for dir in sorted(task_paths):
        print(dir)
        if os.path.isdir(os.path.join(args.task_path, dir)):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

            for trj in sorted(trj_paths):
                print(trj)
                try:
                    os.makedirs(os.path.join(args.task_path,
                                             "video",
                                             dir.split('/')[-1],
                                             trj.split("/")[-1].split(".")[0]))
                except:
                    pass
                saving_dir = os.path.join(args.task_path,
                                          "video",
                                          dir.split('/')[-1],
                                          trj.split("/")[-1].split(".")[0],
                                          trj.split("/")[-1])
                print(f"Saving dir {saving_dir}")
                saving_dir_img = os.path.join(args.task_path,
                                              "img",
                                              dir.split('/')[-1],
                                              trj.split("/")[-1].split(".")[0])
                # logger.debug(f"Trajectory name: {dir}/{trj}")
                i = 0

                with open(trj, "rb") as f:
                    sample = pickle.load(f)
                    logger.debug(f"Sample keys {sample.keys()}")
                    logger.debug(sample)
                    if i == 0:
                        i += 1
                        logger.debug(sample)
                        logger.debug(
                            f"Observation keys: {sample['traj'][0]['obs'].keys()}")
                        logger.debug(
                            f"{sample['traj'][0]['obs']['ee_aa']}")

                    # take the Trajectory obj from the trajectory
                    trajectory_obj = sample['traj']
                    i = 0
                    obj_in_hand = 0
                    start_moving = 0
                    end_moving = 0
                    for t in range(sample['len']):
                        logger.debug(f"Time-step {t}")
                        status = trajectory_obj[t]['info']['status']
                        task_description = f"Trajectory status {status}"
                        # get elements
                        time_step_dict = trajectory_obj[t]
                        obs_t = time_step_dict['obs']
                        # get action
                        try:
                            action_t = trajectory_obj.get(t)['action']
                            for i in range(len(action_t)-1):
                                action_value = action_t[i]
                                if action_value < ranges[i][0]:
                                    ranges[i][0] = action_value
                                elif action_value > ranges[i][1]:
                                    ranges[i][1] = action_value
                        except KeyError:
                            pass
        print(ranges)
