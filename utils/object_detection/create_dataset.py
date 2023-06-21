import os
import os.path as osp
import glob
import pickle as pkl
import numpy as np
from multiprocessing import Pool
import functools
# import mmcv
import cv2
# from mmengine.fileio import dump, load
# from mmengine.utils import track_iter_progress
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from multi_task_il.datasets.multi_task_datasets import MultiTaskPairedDataset
categories = {
    0: "greenbox",
    1: "yellowbox",
    2: "bluebox",
    3: "redbox"
}


def save_json(json_name, dict_list):
    # 2. Write json
    gt_dict = dict()
    gt_dict = {'images': [],
               'annnotations': []
               }
    for key in gt_dict.keys():
        for dict_ in dict_list:
            gt_dict[key] = gt_dict[key] + dict_[key]
    gt_dict['categories'] = [{0: "greenbox",
                              1: "yellowbox",
                              2: "bluebox",
                              3: "redbox"}]

    with open(f'{json_name}', 'w') as outfile:
        json.dump(gt_dict, outfile)


# ---- NOTE ---- #
# bottom_right_corner = upper_left_corner
# upper_left_corner = bottom_right_corner

def create_annotation_dict(task_name: str, traj_index: int, frame_index: int, trj_obj: any, image_save_path: str):
    # get image
    image_id = int('{:03d}'.format(int(task_name.split(
        '_')[-1])+1) + '{:03d}'.format(traj_index+1) + '{:03d}'.format(frame_index+1))
    # print(image_id)
    image = trj_obj.get(
        frame_index)['obs'][f"{camera_name}_image"][:, :, ::-1]
    cv2.imwrite(os.path.join(
        image_save_path, f"{frame_index}.jpg"), image)

    # ---- Fill the annotation dict ---#
    # 1. File dict
    file_dict = dict()
    file_dict['file_name'] = os.path.join(
        image_save_path, f"{frame_index}.jpg")
    file_dict['height'] = image.shape[0]
    file_dict['width'] = image.shape[1]
    file_dict['id'] = image_id
    # 2. Annotations
    target_obj_id = trj_obj.get(
        frame_index)['obs']['target-object']
    annotation_dict = dict()
    annotation_id = int('{:03d}'.format(int(task_name.split(
        '_')[-1])+1) + '{:03d}'.format(traj_index+1) + '{:03d}'.format(frame_index+1) + '{:03d}'.format(0+1))

    annotation_dict['image_id'] = image_id
    annotation_dict['id'] = annotation_id
    annotation_dict['category_id'] = target_obj_id
    top_left_x = trj_obj.get(
        frame_index)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['bottom_right_corner'][0]
    top_left_y = trj_obj.get(
        frame_index)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['bottom_right_corner'][1]
    # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
    bottom_right_x = trj_obj.get(
        frame_index)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['upper_left_corner'][0]
    bottom_right_y = trj_obj.get(
        frame_index)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['upper_left_corner'][1]
    # print(
    #     f"Bottom Right X {bottom_right_x} - Bottom Right Y {bottom_right_y}")

    # bounding-box
    # right_x - left_x
    width = bottom_right_x - top_left_x
    # left_y - right_y
    height = bottom_right_y - top_left_y
    annotation_dict['bbox'] = [
        top_left_x, top_left_y, width, height]
    annotation_dict['iscrowd'] = 0
    # print(annotation_dict['bbox'])

    return file_dict, annotation_dict


def convert_traj_to_coco(save_folder: str = None, camera_name: str = "camera_front", num_workers: int = 1, frame_to_save: int = 1, task_path: str = None):
    print(task_path)
    gt_dict = {'images': [],
               'annnotations': []}

    # For the current task iterate over all the trajectories
    if ".pkl" in task_path:
        # 1. Create folder for current task
        task_name = task_path.split('/')[-2]
        traj_index = int(task_path.split(
            '/')[-1].split('.pkl')[0].split('traj')[-1])
        save_path = os.path.join(save_folder, task_name)
        try:
            os.makedirs(save_path)
        except:
            pass
        trj = task_path
        # Create folder for the current trajectory
        trj_name = trj.split('/')[-1].split(".")[0]
        image_save_path = os.path.join(
            save_folder, task_name, trj_name, 'image')
        try:
            os.makedirs(image_save_path)
        except:
            pass

        # open pkl file
        with open(trj, "rb") as f:
            sample = pkl.load(f)
            trj_obj = sample['traj']
            file_dict, annotation_dict = create_annotation_dict(
                task_name=task_name,
                traj_index=traj_index,
                frame_index=frame_to_save,
                trj_obj=trj_obj,
                image_save_path=image_save_path
            )
            # append dicts
            gt_dict['images'].append(file_dict)
            gt_dict['annnotations'].append(annotation_dict)

            # image_rgb = cv2.circle(
            #     np.array(image), (top_left_x, top_left_y), radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite(f"provola_{t}.png", image_rgb)

    else:
        # 1. Create folder for current task
        task_name = task_path.split('/')[-1]
        save_path = os.path.join(save_folder, task_name)
        try:
            os.makedirs(save_path)
        except:
            pass
        for i, trj in enumerate(sorted(task_path)):
            # Create folder for the current trajectory
            trj_name = trj.split('/')[-1].split(".")[0]
            image_save_path = os.path.join(
                save_folder, task_name, trj_name, 'image')
            try:
                os.makedirs(image_save_path)
            except:
                pass

            # open pkl file
            with open(trj, "rb") as f:
                sample = pkl.load(f)
                trj_obj = sample['traj']

                for t in range(len(trj_obj)):

                    file_dict, annotation_dict = create_annotation_dict(
                        task_name=task_name,
                        traj_index=i,
                        frame_index=t,
                        trj_obj=trj_obj,
                        image_save_path=image_save_path
                    )
                    # append dicts
                    gt_dict['images'].append(file_dict)
                    gt_dict['annnotations'].append(annotation_dict)

                    # image_rgb = cv2.circle(
                    #     np.array(image), (top_left_x, top_left_y), radius=1, color=(0, 0, 255), thickness=-1)
                    # cv2.imwrite(f"provola_{t}.png", image_rgb)
    return gt_dict


def load_trajectories(conf_file, mode='train'):
    conf_file.dataset_cfg.mode = mode
    return hydra.utils.instantiate(conf_file.dataset_cfg)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default="/")
    parser.add_argument('--trj_path', default="/")
    parser.add_argument('--task_name', default="/")
    parser.add_argument('--robot_name', default="/")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument(
        "--camera_name",  default="camera_front")
    parser.add_argument("--debug", action='store_true',
                        help="whether or not attach the debugger")
    parser.add_argument("--depth", action='store_true',
                        help="whether or not render depth")
    parser.add_argument("--split_trainig_validation", action='store_true')
    parser.add_argument("--model", default="/")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    save_folder = args.save_folder
    trj_path = args.trj_path
    task_name = args.task_name
    robot_name = args.robot_name
    camera_name = args.camera_name

    try:
        os.makedirs(os.path.join(save_folder, task_name,
                    f"{robot_name}_{task_name}"))
    except:
        pass

    if not args.split_trainig_validation:
        # 1. Load .pkl files
        task_paths = sorted(glob.glob(os.path.join(
            args.trj_path, f"{robot_name}_{task_name}", 'task_*')))

        f = functools.partial(convert_traj_to_coco,
                              os.path.join(save_folder, task_name,
                                           f"{robot_name}_{task_name}"),
                              camera_name,
                              args.num_workers,
                              1
                              )
        parallel = args.num_workers > 1
        if parallel:
            with Pool(args.num_workers) as p:
                gt_dict_list = p.map(f, task_paths)
        else:
            for task_path in task_paths:
                gt_dict_list = [f(task_path)]

        save_json(json_name=f"{robot_name}_full_dataset",
                  dict_list=gt_dict_list)

    else:
        # 1. Create Multi-Task dataset to get training indices
        # load configuration file
        conf_file = OmegaConf.load(os.path.join(args.model, "config.yaml"))
        training_dataset = load_trajectories(conf_file=conf_file,
                                             mode='train')
        validation_dataset = load_trajectories(conf_file=conf_file,
                                               mode='val')

        # ---- TRAINING ---- #
        training_files = training_dataset.agent_files
        trajectories_path = []
        for task in training_files.keys():
            for variation in training_files[task].keys():
                for trj in training_files[task][variation]:
                    trajectories_path.append(trj)

        f = functools.partial(convert_traj_to_coco,
                              os.path.join(save_folder, task_name,
                                           f"{robot_name}_{task_name}"),
                              camera_name,
                              args.num_workers,
                              1
                              )
        parallel = args.num_workers > 1
        if parallel:
            with Pool(args.num_workers) as p:
                gt_dict_list = p.map(f, trajectories_path)
        else:
            for task_path in trajectories_path:
                gt_dict_list = [f(task_path)]

        save_json(json_name=f"{robot_name}_train.json",
                  dict_list=gt_dict_list)

        # ---- VALIDATION ---- #
        validation_files = validation_dataset.agent_files
        trajectories_path = []
        for task in validation_files.keys():
            for variation in validation_files[task].keys():
                for trj in validation_files[task][variation]:
                    trajectories_path.append(trj)

        f = functools.partial(convert_traj_to_coco,
                              os.path.join(save_folder, task_name,
                                           f"{robot_name}_{task_name}"),
                              camera_name,
                              args.num_workers,
                              1
                              )
        parallel = args.num_workers > 1
        if parallel:
            with Pool(args.num_workers) as p:
                gt_dict_list = p.map(f, trajectories_path)
        else:
            for task_path in trajectories_path:
                gt_dict_list = [f(task_path)]

        save_json(json_name=f"{robot_name}_val.json",
                  dict_list=gt_dict_list)
