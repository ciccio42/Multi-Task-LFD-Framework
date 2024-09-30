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
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

MAX_DIST = 10
MIN_DIST = 0.01

# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}
# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
# object_to_id = {"round-nut": 0, "round-nut-2": 1, "round-nut-3": 2}
TASK_NAME = "pick_place"

RANGES = np.array(
    [[-0.3, 0.3], [-0.3, 0.3], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])


def denormalize_action(norm_action):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(RANGES.shape[0]):
        action[d] = (0.5 * (action[d] + 1) *
                     (RANGES[d, 1] - RANGES[d, 0])) + RANGES[d, 0]
    return action


def adjust_bb(bb, crop_params=[0, 30, 120, 120]):

    x1_old, y1_old, x2_old, y2_old = bb
    x1_old = int(x1_old)
    y1_old = int(y1_old)
    x2_old = int(x2_old)
    y2_old = int(y2_old)

    top, left = crop_params[0], crop_params[2]
    img_height, img_width = 376, 672
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    # Modify bb based on computed resized-crop
    # 1. Take into account crop and resize
    x_scale = 180/box_w
    y_scale = 100/box_h
    x1 = int((x1_old/x_scale)+left)
    x2 = int((x2_old/x_scale)+left)
    y1 = int((y1_old/y_scale)+top)
    y2 = int((y2_old/y_scale)+top)
    return [x1, y1, x2, y2]


def build_tvf_formatter():
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """

    crop_params = [20, 25, 80, 75]  # [30, 0, 35, 35]  # [20, 25, 80, 75]
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]
    height = 200
    width = 360

    def resize_crop(img_name, img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        obs = ToTensor()(img.copy())
        # [:,:,::-1]
        cv2.imwrite(f"{img_name}.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width), antialias=True)
        cv2.imwrite(f"{img_name}_cropped.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        # obs = Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])(obs)
        cv2.imwrite(f"{img_name}_normalized.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)

        return obs
    return resize_crop


def create_video_writer(pickle_file_path, camera_name, depth, img_width, img_height):
    # create video Write
    video_name_rgb = pickle_file_path.split(
        ".pkl")[0] + f"{camera_name}_rgb.avi"
    video_rgb = cv2.VideoWriter(
        video_name_rgb, fourcc, 20, (img_width, img_height))

    if depth:
        video_name_depth = pickle_file_path.split(
            ".pkl")[0] + f"{camera_name}_depth.avi"
        video_depth = cv2.VideoWriter(video_name_depth, cv2.VideoWriter_fourcc(
            *'MP42'), 20, (img_width, img_height), False)
        return video_rgb, video_depth

    return video_rgb, None


def normalize_depth(depth_img, obs):
    near = obs['znear']
    far = 5  # obs['zfar']
    depth_norm = (depth_img-near)/(far-near)
    return depth_norm


def write_frame(rgb_video_writer, depth_video_writer, camera_name, obs, str, task_id):
    # get robot agent view
    if camera_name == "image":
        image_rgb = np.array(obs[f'image'][..., ::-1])
        if str:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.25
            thickness = 1
            cv2.putText(image_rgb, str, (0, 99), font, font_scale,
                        (0, 255, 0), thickness, cv2.LINE_AA)
    else:
        
        if obs.get(f'{camera_name}_image_full_size', None) is not None:
            image = cv2.imdecode(obs[f"{camera_name}_image_full_size"], cv2.IMREAD_COLOR) 
        else:
            image = obs[f"{camera_name}_image"]
        
        cv2.imwrite("original.png", image)
        cv2.imwrite("original_rgb.png", image[..., ::-1])
        image_rgb = np.array(image).copy()

    
    if obs.get('predicted_bb') is not None:
        predicted_bb_dict = obs['predicted_bb']
        camera_names = ['camera_front']
        for camera_name in camera_names:
            predicted_bbs = predicted_bb_dict[camera_name]
            for indx, predicted_bb in enumerate(predicted_bbs):
                # print(predicted_bb)
                
                adjusted_bb = adjust_bb(bb=predicted_bb)

                if indx == 0:
                    color = (0, 255, 0)
                elif indx == 1:
                    color = (0, 255, 0)
                
                image_rgb = cv2.rectangle(
                    image_rgb, 
                    (adjusted_bb[0], adjusted_bb[1]),
                    (adjusted_bb[2], adjusted_bb[3]), 
                    color, 
                    1)    
                
    #     center = target_obj_bb['center']
    #     upper_left_corner = target_obj_bb['upper_left_corner']
    #     bottom_right_corner = target_obj_bb['bottom_right_corner']
    #     image_rgb = cv2.circle(
    #         image_rgb, center, radius=1, color=(0, 0, 255), thickness=-1)
    #     image_rgb = cv2.rectangle(
    #         image_rgb, upper_left_corner,
    #         bottom_right_corner, (255, 0, 0), 1)

    #     if 'pick_place' in TASK_NAME:
    #         try:
    #             bin_bb = obs['obj_bb'][camera_name.split('_image')[
    #                 0]]['bin']
    #             center = bin_bb['center']
    #             upper_left_corner = bin_bb['upper_left_corner']
    #             bottom_right_corner = bin_bb['bottom_right_corner']
    #             # image_rgb = cv2.circle(
    #             #     image_rgb, center, radius=1, color=(0, 0, 255), thickness=-1)
    #             # image_rgb = cv2.rectangle(
    #             #     image_rgb, upper_left_corner,
    #             #     bottom_right_corner, (255, 0, 0), 1)
    #         except:
    #             pass

    #     cv2.imwrite("box.png", image_rgb)

    rgb_video_writer.write(image_rgb)



TASK_DICT = OrderedDict()
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
    img_formatter = build_tvf_formatter()

    average_frame = 0
    number_task = 0
    for dir in sorted(task_paths):
        print(dir)
        task_id = int(dir.split('/')[-1].split('task_')[-1].lstrip())
        print(f"Task id {task_id}")
        if task_id == '':
            task_id = 0
        TASK_DICT[args.task_path] = OrderedDict()
        if os.path.isdir(os.path.join(args.task_path, dir)):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

            TASK_DICT[args.task_path][dir] = 0.0
            for j, trj in enumerate(sorted(trj_paths)):
                # if j == 0 or j == 10:
                if  True:
                    with open(trj, "rb") as f:
                        sample = pickle.load(f)
                        TASK_DICT[args.task_path][dir] += len(sample)
                    
                    print(trj)

                    os.makedirs(os.path.join(args.task_path,
                                             "video",
                                             dir.split('/')[-1],
                                             trj.split("/")[-1].split(".")[0]), exist_ok=True)

                    saving_dir = os.path.join(args.task_path,
                                              "video",
                                              dir.split('/')[-1],
                                              trj.split(
                                                  "/")[-1].split(".")[0],
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
                        if i == 0:
                            i += 1
                            logger.debug(sample)
                            logger.debug(
                                f"Observation keys: {sample[0]['obs'].keys()}")
                            # logger.debug(
                            #     f"{sample[0]['obs']['ee_aa']}")

                            front_video_rgb = None
                            right_video_rgb = None
                            left_video_rgb = None
                            if 'camera_front_image' in sample.get(
                                    0)['obs'].keys():
                                if 'camera_front_image_full_size' in sample.get(
                                    0)['obs'].keys():
                                    image = cv2.imdecode(sample.get(0)['obs']['camera_front_image_full_size'], cv2.IMREAD_COLOR)
                                    img_width = image.shape[1]
                                    img_height = image.shape[0]
                                else:
                                    img_width = sample.get(
                                        0)['obs']['camera_front_image'].shape[1]
                                    img_height = sample.get(
                                        0)['obs']['camera_front_image'].shape[0]
                                print(img_width, img_height)
                                front_video_rgb, front_video_depth = create_video_writer(
                                    saving_dir, "camera_front", depth=args.depth, img_width=img_width, img_height=img_height)
                                # right_video_rgb, right_video_depth = create_video_writer(
                                #     saving_dir, "camera_lateral_right", depth=args.depth, img_width=img_width, img_height=img_height)
                                # left_video_rgb, left_video_depth = create_video_writer(
                                #     saving_dir, "camera_lateral_left", depth=args.depth, img_width=img_width, img_height=img_height)
                                # eye_in_hand_rgb, eye_in_hand_depth = create_video_writer(
                                #     saving_dir, "camera_robot", depth=args.depth, img_width=img_width, img_height=img_height)
                            else:
                                img_width = sample.get(
                                    0)['obs']['image'].shape[1]
                                img_height = sample.get(
                                    0)['obs']['image'].shape[0]
                                print(img_width, img_height)
                                front_video_rgb, front_video_depth = create_video_writer(
                                    saving_dir, "image", depth=args.depth, img_width=img_width, img_height=img_height)

                        # take the Trajectory obj from the trajectory
                        trajectory_obj = sample
                        i = 0
                        obj_in_hand = 0
                        start_moving = 0
                        end_moving = 0
                        for t in range(len(trajectory_obj)):
                            logger.debug(f"Time-step {t}")
                            # task description
                            # TASK_NAME = args.task_path.split('/')[-2]
                            # sub_task =  dir.split("_")[-1]
                            # if t != 0 and 'status' in trajectory_obj.get(t)['info'].keys():
                            #     status = trajectory_obj[t]['info']['status']
                            # else:
                            #     status = None
                            status = None
                            task_description = f"Trajectory status {status}"
                            # get elements
                            time_step_dict = trajectory_obj[t]
                            obs_t = time_step_dict['obs']
                            if 'camera_front_image' in sample.get(
                                    0)['obs'].keys():
                                write_frame(
                                    front_video_rgb, front_video_depth, 'camera_front', obs_t, task_description, task_id)
                                # write_frame(right_video_rgb, right_video_depth,
                                #             "camera_lateral_right", obs_t, task_description)
                                # write_frame(
                                #     left_video_rgb, left_video_depth, "camera_lateral_left", obs_t, task_description)
                                # write_frame(
                                #     eye_in_hand_rgb, eye_in_hand_depth, "eye_in_hand", obs_t, task_description)
                            else:
                                pass
                                # write_frame(
                                #     front_video_rgb, front_video_depth, 'image', obs_t, task_description)
                            # get action
                            try:
                                action_t = trajectory_obj.get(t)['action']

                                # print(f"Norm action{action_t[:3]}")
                                # print(
                                #     f"Denorm action {denormalize_action(action_t)[:3]}")
                            except KeyError:
                                pass

                            if t == 1:
                                try:
                                    os.makedirs(saving_dir_img)
                                except:
                                    pass
                                #
                                if  trajectory_obj[t]['obs'].get('camera_front_image_full_size', None) is not None:
                                    obs = cv2.imdecode(trajectory_obj[t]['obs']['camera_front_image_full_size'], cv2.IMREAD_COLOR)
                                else:
                                    obs = trajectory_obj[t]['obs']['camera_front_image']
                                img_name = os.path.join(
                                    saving_dir_img, f"{t}")
                                img_formatter(img_name, obs)
                            if t == len(trajectory_obj)-1:
                                if  trajectory_obj[t]['obs'].get('camera_front_image_full_size', None)  is not None:
                                    obs = cv2.imdecode(trajectory_obj[t]['obs']['camera_front_image_full_size'], cv2.IMREAD_COLOR)
                                else:
                                    obs = trajectory_obj[t]['obs']['camera_front_image']
                                img_name = os.path.join(
                                    saving_dir_img, f"{t}")
                                img_formatter(img_name, obs)
                            # if t != 0 and 'status' in trajectory_obj.get(t)['info'].keys():
                            #     if trajectory_obj.get(t)['info']['status'] == 'obj_in_hand' and obj_in_hand == 0:
                            #         obj_in_hand = 1
                            #         #
                            #         obs = trajectory_obj[t]['obs']['camera_front_image']
                            #         img_name = os.path.join(
                            #             saving_dir_img, f"{t}")
                            #         img_formatter(img_name, obs)
                            #     if trajectory_obj.get(t)['info']['status'] == 'moving' and start_moving == 0:
                            #         start_moving = t
                            #     elif trajectory_obj.get(t)['info']['status'] != 'moving' and start_moving != 0 and end_moving == 0:
                            #         end_moving = t
                            #         middle_moving_t = start_moving + \
                            #             int((end_moving-start_moving)/2)
                            #         obs = trajectory_obj.get(middle_moving_t)[
                            #             'obs']['camera_front_image']
                            #         img_name = os.path.join(
                            #             saving_dir_img, f"{middle_moving_t}")
                            #         img_formatter(img_name, obs)
                            #     if t == len(trajectory_obj)-1:
                            #         #
                            #         obs = trajectory_obj[t]['obs']['camera_front_image']#[..., ::-1]
                            #         img_name = os.path.join(
                            #             saving_dir_img, f"{t}")
                            #         img_formatter(img_name, obs)
                        # cv2.destroyAllWindows()
                        front_video_rgb.release()
                        if right_video_rgb != None:
                            right_video_rgb.release()
                            left_video_rgb.release()

            TASK_DICT[args.task_path][dir] = TASK_DICT[args.task_path][dir] / \
                len(trj_paths)
            average_frame += TASK_DICT[args.task_path][dir]
            number_task += 1

        print(
            f"Task {args.task_path} - Average frame {average_frame/number_task}")
