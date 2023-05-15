import os
import os.path as osp
import glob
import pickle as pkl
import numpy as np
from multiprocessing import Pool
import functools
import mmcv
import cv2
from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

categories = {
    0: "greenbox",
    1: "yellowbox",
    2: "bluebox",
    3: "redbox"
}


# ---- NOTE ---- #
# bottom_right_corner = upper_left_corner
# upper_left_corner = bottom_right_corner

def convert_traj_to_coco(save_folder: str = None, camera_name: str = "camera_front", num_workers: int = 1, task_path: str = None):
    print(task_path)
    # 1. Create folder for current task
    task_name = task_path.split('/')[-1]
    save_path = os.path.join(save_folder, task_name)
    try:
        os.makedirs(save_path)
    except:
        pass

    gt_dict = {'images': [],
               'annnotations': []}

    # For the current task iterate over all the trajectories
    trj_paths = sorted(glob.glob(os.path.join(task_path, 'traj*.pkl')))
    for i, trj in enumerate(sorted(trj_paths)):
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
                # get image
                image_id = '{:03d}'.format(int(task_name.split(
                    '_')[-1])) + '{:03d}'.format(i) + '{:03d}'.format(t)
                print(image_id)
                image = trj_obj.get(
                    t)['obs'][f"{camera_name}_image"][:, :, ::-1]
                cv2.imwrite(os.path.join(image_save_path, f"{t}.jpg"), image)

                # ---- Fill the annotation dict ---#
                # 1. File dict
                file_dict = dict()
                file_dict['file_name'] = os.path.join(
                    image_save_path, f"{t}.jpg")
                file_dict['height'] = image.shape[0]
                file_dict['width'] = image.shape[1]
                file_dict['id'] = int(image_id)
                # 2. Annotations
                target_obj_id = trj_obj.get(
                    t)['obs']['target-object']
                annotation_dict = dict()
                annotation_id = '{:03d}'.format(int(task_name.split(
                    '_')[-1])) + '{:03d}'.format(i) + '{:03d}'.format(t) + '{:03d}'.format(0)

                annotation_dict['image_id'] = int(image_id)
                annotation_dict['id'] = int(annotation_id)
                annotation_dict['category_id'] = target_obj_id
                top_left_x = trj_obj.get(
                    t)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['bottom_right_corner'][0]
                top_left_y = trj_obj.get(
                    t)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['bottom_right_corner'][1]
                # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
                bottom_right_x = trj_obj.get(
                    t)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['upper_left_corner'][0]
                bottom_right_y = trj_obj.get(
                    t)['obs']['obj_bb'][camera_name][categories[target_obj_id]]['upper_left_corner'][1]
                # print(
                #     f"Bottom Right X {bottom_right_x} - Bottom Right Y {bottom_right_y}")

                # bounding-box
                # right_x - left_x
                width = bottom_right_x - top_left_x
                # left_y - right_y
                height = bottom_right_y - top_left_y
                annotation_dict['bbox'] = [
                    top_left_x, top_left_y, width, height]
                print(annotation_dict['bbox'])
                annotation_dict['iscrowd'] = 0

                # append dicts
                gt_dict['images'].append(file_dict)
                gt_dict['annnotations'].append(annotation_dict)

                # image_rgb = cv2.circle(
                #     np.array(image), (top_left_x, top_left_y), radius=1, color=(0, 0, 255), thickness=-1)
                # cv2.imshow("provola", np.array(image_rgb))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    return gt_dict


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
    args = parser.parse_args()

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

    # 1. Load .pkl files
    task_paths = glob.glob(os.path.join(
        args.trj_path, f"{robot_name}_{task_name}", 'task_*'))

    print(args.num_workers)
    f = functools.partial(convert_traj_to_coco,
                          os.path.join(save_folder, task_name,
                                       f"{robot_name}_{task_name}"),
                          camera_name,
                          args.num_workers
                          )
    parallel = args.num_workers > 1
    if parallel:
        with Pool(args.num_workers) as p:
            p.map(f, task_paths)
    else:
        for task_path in task_paths:
            [f(task_path)]
    # for dir in sorted(task_paths):
    #     print(dir)
    #     if os.path.isdir(dir):
    #         trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

    #         print(trj_paths)
