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

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

MAX_DIST = 10
MIN_DIST = 0.01

# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

RANGES = np.array(
    [[-0.3, 0.3], [-0.3, 0.3], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])


def denormalize_action(norm_action):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(RANGES.shape[0]):
        action[d] = (0.5 * (action[d] + 1) *
                     (RANGES[d, 1] - RANGES[d, 0])) + RANGES[d, 0]
    return action


def build_tvf_formatter():
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """

    crop_params = [30, 0, 35, 35]  # [20, 25, 80, 75]
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]
    height = 100
    width = 180

    def resize_crop(img_name, img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]

        obs = ToTensor()(img.copy())
        cv2.imwrite(f"{img_name}.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width), antialias=True)
        cv2.imwrite(f"{img_name}_cropped.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        obs = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(obs)
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


def write_frame(rgb_video_writer, depth_video_writer, camera_name, obs, str):
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
        image_rgb = np.array(obs[f"{camera_name}_image"][:, :, ::-1])
        # if camera_name == 'camera_lateral_right':
        #     cv2.imshow(f"'camera_front_image'", image_rgb)
        #     cv2.waitKey(100)
        #     cv2.destroyAllWindows()
        # if str:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.3
        #     thickness = 1
        #     cv2.putText(image_rgb, str, (0, 99), font, font_scale,
        #                 (0, 255, 0), thickness, cv2.LINE_AA)

    if 'target-object' in obs.keys() and (camera_name != 'eye_in_hand' and camera_name != 'image'):
        target_obj_id = obs['target-object']
        target_obj_bb = None
        for object_names in object_to_id.keys():
            if target_obj_id == object_to_id[object_names]:
                target_obj_bb = obs['obj_bb'][camera_name.split('_image')[
                    0]][object_names]

        center = target_obj_bb['center']
        upper_left_corner = target_obj_bb['upper_left_corner']
        bottom_right_corner = target_obj_bb['bottom_right_corner']
        image_rgb = cv2.circle(
            image_rgb, center, radius=1, color=(0, 0, 255), thickness=-1)
        image_rgb = cv2.rectangle(
            image_rgb, upper_left_corner,
            bottom_right_corner, (255, 0, 0), 1)
        # cv2.imshow(camera_name, image_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    rgb_video_writer.write(image_rgb)

    if depth_video_writer is not None:
        image_depth = obs[f'{camera_name}_depth']
        # image_depth_norm = obs[f'{camera_name}_depth_norm']
        # depth_norm = normalize_depth(image_depth, obs)
        # assert np.all(depth_norm >= 0.0) and np.all(depth_norm <= 1.0)
        # depth_video_writer.write((depth_norm*255).astype(np.uint8))
        depth_video_writer.write(image_depth)


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

    for dir in sorted(task_paths):
        print(dir)
        if os.path.isdir(os.path.join(args.task_path, dir)) and ('_00'):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))

            for j, trj in enumerate(sorted(trj_paths)):
                if j == 0:
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

                            front_video_rgb = None
                            right_video_rgb = None
                            left_video_rgb = None
                            if 'camera_front_image' in sample['traj'].get(
                                    0)['obs'].keys():
                                img_width = sample['traj'].get(
                                    0)['obs']['camera_front_image'].shape[1]
                                img_height = sample['traj'].get(
                                    0)['obs']['camera_front_image'].shape[0]
                                print(img_width, img_height)
                                front_video_rgb, front_video_depth = create_video_writer(
                                    saving_dir, "camera_front", depth=args.depth, img_width=img_width, img_height=img_height)
                                right_video_rgb, right_video_depth = create_video_writer(
                                    saving_dir, "camera_lateral_right", depth=args.depth, img_width=img_width, img_height=img_height)
                                left_video_rgb, left_video_depth = create_video_writer(
                                    saving_dir, "camera_lateral_left", depth=args.depth, img_width=img_width, img_height=img_height)
                                eye_in_hand_rgb, eye_in_hand_depth = create_video_writer(
                                    saving_dir, "camera_robot", depth=args.depth, img_width=img_width, img_height=img_height)
                            else:
                                img_width = sample['traj'].get(
                                    0)['obs']['image'].shape[1]
                                img_height = sample['traj'].get(
                                    0)['obs']['image'].shape[0]
                                print(img_width, img_height)
                                front_video_rgb, front_video_depth = create_video_writer(
                                    saving_dir, "image", depth=args.depth, img_width=img_width, img_height=img_height)

                        # take the Trajectory obj from the trajectory
                        trajectory_obj = sample['traj']
                        i = 0
                        obj_in_hand = 0
                        start_moving = 0
                        end_moving = 0
                        for t in range(len(trajectory_obj)):
                            logger.debug(f"Time-step {t}")
                            # task description
                            # task_name = args.task_path.split('/')[-2]
                            # sub_task =  dir.split("_")[-1]
                            if t != 0 and 'status' in trajectory_obj.get(t)['info'].keys():
                                status = trajectory_obj[t]['info']['status']
                            else:
                                status = None
                            task_description = f"Trajectory status {status}"
                            # get elements
                            time_step_dict = trajectory_obj[t]
                            obs_t = time_step_dict['obs']
                            # logger.info(obs_t['gripper_qpos'])
                            # cv2.imshow(f"'camera_front_image'",
                            #            obs_t['camera_front_image'][:, :, ::-1])
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            if 'camera_front_image' in sample['traj'].get(
                                    0)['obs'].keys():
                                write_frame(
                                    front_video_rgb, front_video_depth, 'camera_front', obs_t, task_description)
                                write_frame(right_video_rgb, right_video_depth,
                                            "camera_lateral_right", obs_t, task_description)
                                write_frame(
                                    left_video_rgb, left_video_depth, "camera_lateral_left", obs_t, task_description)
                                write_frame(
                                    eye_in_hand_rgb, eye_in_hand_depth, "eye_in_hand", obs_t, task_description)
                            else:
                                pass
                                # write_frame(
                                #     front_video_rgb, front_video_depth, 'image', obs_t, task_description)
                            # get action
                            try:
                                action_t = trajectory_obj.get(t)['action']

                                print(f"Norm action{action_t[:3]}")
                                print(
                                    f"Denorm action {denormalize_action(action_t)[:3]}")
                                # logger.info(
                                #     f"Action at time-step {t}: {action_t}")
                            except KeyError:
                                pass

                            if t == 1:
                                try:
                                    os.makedirs(saving_dir_img)
                                except:
                                    pass
                                obs = trajectory_obj[t]['obs']['image'][:, :, ::-1]
                                img_name = os.path.join(
                                    saving_dir_img, f"{t}")
                                img_formatter(img_name, obs)
                            if t != 0 and 'status' in trajectory_obj.get(t)['info'].keys():
                                if trajectory_obj.get(t)['info']['status'] == 'obj_in_hand' and obj_in_hand == 0:
                                    obj_in_hand = 1
                                    obs = trajectory_obj[t]['obs']['image'][:, :, ::-1]
                                    img_name = os.path.join(
                                        saving_dir_img, f"{t}")
                                    img_formatter(img_name, obs)
                                if trajectory_obj.get(t)['info']['status'] == 'moving' and start_moving == 0:
                                    start_moving = t
                                elif trajectory_obj.get(t)['info']['status'] != 'moving' and start_moving != 0 and end_moving == 0:
                                    end_moving = t
                                    middle_moving_t = start_moving + \
                                        int((end_moving-start_moving)/2)
                                    obs = trajectory_obj.get(middle_moving_t)[
                                        'obs']['image'][:, :, ::-1]
                                    img_name = os.path.join(
                                        saving_dir_img, f"{middle_moving_t}")
                                    img_formatter(img_name, obs)
                                if t == len(trajectory_obj)-1:
                                    obs = trajectory_obj[t]['obs']['image'][:, :, ::-1]
                                    img_name = os.path.join(
                                        saving_dir_img, f"{t}")
                                    img_formatter(img_name, obs)
                        cv2.destroyAllWindows()
                        front_video_rgb.release()
                        if right_video_rgb != None:
                            right_video_rgb.release()
                            left_video_rgb.release()
