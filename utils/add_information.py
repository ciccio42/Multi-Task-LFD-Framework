from multi_task_il.datasets.savers import _compress_obs
import pickle
import os
import cv2
import debugpy
import logging
from PIL import Image
import numpy as np
import random
import glob
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop
from collections import OrderedDict
import copy
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

IMG_HEIGHT = 376
IMG_WIDTH = 672


def overwrite_pkl_file(pkl_file_path, sample):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['camera_front_image']
            # _img =
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['camera_front_image'] = im_string
            obs = traj.get(t)['obs']

        obs = _compress_obs(obs)
        traj.change_obs(t, obs)
        logger.debug(obs.keys())

    pickle.dump({
        'traj': traj,
        'len': len(traj),
        'env_type': sample['env_type'],
        'task_id': sample['task_id']}, open(pkl_file_path, 'wb'))


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

    task_paths = glob.glob(os.path.join(args.task_path, 'task_*'))

    for dir in sorted(task_paths):
        print(dir)
        task_number = int(dir.split("_")[-1])
        target_object = int(task_number / 4)
        target_box_id = task_number % 4
        print(f"Target-obj-id {target_object} - Target-box-id {target_box_id}")
        if os.path.isdir(os.path.join(args.task_path, dir)):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

            for trj in tqdm(sorted(trj_paths)):
                if True:
                    with open(trj, "rb") as f:
                        sample = pickle.load(f)

                    for t in range(len(sample['traj'])):
                        obs = sample['traj'].get(t)['obs']
                        obs['target-object'] = target_object
                        obs['target-box-id'] = target_box_id

                        if args.debug:
                            for camera_name in ["camera_front", "camera_lateral_right", "camera_lateral_left", "eye_in_hand"]:
                                if len(obs[f"{camera_name}_image"].shape) != 3:
                                    img = cv2.imdecode(
                                        obs[f"{camera_name}_image"], cv2.IMREAD_COLOR)
                                else:
                                    img = obs[f"{camera_name}_image"]
                                cv2.imwrite("prova.jpg", img)

                    overwrite_pkl_file(pkl_file_path=trj,
                                       sample=sample)
