import pickle, os
import cv2
import debugpy
import logging
from PIL import Image
import numpy as np
import random

logging.basicConfig(format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

MAX_DIST = 10
MIN_DIST = 0.01

def create_video_writer(pickle_file_path, camera_name, depth):
    # create video Write
    video_name_rgb = pickle_file_path.split(".pkl")[0] + f"{camera_name}_rgb.avi"
    video_rgb = cv2.VideoWriter(video_name_rgb, fourcc, 20, (args.obs_width,args.obs_height))
    
    if depth:
        video_name_depth = pickle_file_path.split(".pkl")[0] + f"{camera_name}_depth.avi"
        video_depth = cv2.VideoWriter(video_name_depth, cv2.VideoWriter_fourcc(*'MP42'), 20, (args.obs_width,args.obs_height), False)
        return video_rgb, video_depth 
    
    return video_rgb, None

def normalize_depth(depth_img, obs):
    near = obs['znear']
    far = 5#obs['zfar']
    depth_norm = (depth_img-near)/(far-near)
    return depth_norm

def write_frame(rgb_video_writer, depth_video_writer, camera_name, obs):
    # get robot agent view
    if camera_name == "image":
        image_rgb = obs[f'image'][...,::-1]
    else:
        image_rgb = obs[f'{camera_name}_image'][...,::-1]
    rgb_video_writer.write(image_rgb)
    if depth_video_writer is not None:
        image_depth = obs[f'{camera_name}_depth']
        image_depth_norm = obs[f'{camera_name}_depth_norm']
        depth_norm = normalize_depth(image_depth, obs)
        assert np.all(depth_norm >= 0.0) and np.all(depth_norm <= 1.0)
        depth_video_writer.write((depth_norm*255).astype(np.uint8))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")    
    parser.add_argument('--obs_height', default=480, type=int, help="Image observation height")    
    parser.add_argument('--obs_width', default=720, type=int, help="Image observation width")    
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--depth", action='store_true', help="whether or not render depth")        
    args = parser.parse_args()
    
    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    logger.info(f"Task path: {args.task_path}")
    i = 0
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    for dir in os.listdir(args.task_path):
        print(dir)
        if os.path.isdir(os.path.join(args.task_path, dir)):
            if dir.split("_")[0] == "task":
                trjs = os.listdir(os.path.join(args.task_path, dir))
                assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
                for trj in trjs:
                    if not os.path.isdir(os.path.join(args.task_path, dir, trj)) and trj.split(".")[1]=="pkl":
                        try:
                            os.makedirs(os.path.join(args.task_path, "video", dir, trj.split(".")[0]))
                        except:
                            pass
                        saving_dir = os.path.join(args.task_path, "video", dir, trj.split(".")[0], trj)
                        logger.debug(f"Trajectory name: {dir}/{trj}")
                        front_video_rgb, front_video_depth = create_video_writer(saving_dir, "image", depth=args.depth) 
                        #right_video_rgb, right_video_depth = create_video_writer(saving_dir, "camera_lateral_right") 
                        #left_video_rgb, left_video_depth = create_video_writer(saving_dir, "camera_lateral_left") 
                        #eye_in_hand_rgb, eye_in_hand_depth = create_video_writer(saving_dir, "eye_in_hand")
                        i = 0 
                        pickle_file_path = os.path.join(args.task_path, dir, trj)
                        with open(pickle_file_path, "rb") as f:
                            sample = pickle.load(f)
                            logger.debug(f"Sample keys {sample.keys()}")
                            logger.debug(sample)
                            if i == 0:
                                i += 1
                                logger.debug(sample)
                                logger.debug(f"Observation keys: {sample['traj'][0]['obs'].keys()}")


                            # take the Trajectory obj from the trajectory
                            trajectory_obj = sample['traj']
                            i = 0
                            for t in range(sample['len']):
                                logger.debug(f"Time-step {t}")
                                # get elements
                                time_step_dict = trajectory_obj[t]
                                obs_t = time_step_dict['obs']
                                write_frame(front_video_rgb, front_video_depth, "image", obs_t)
                                #write_frame(right_video_rgb, right_video_depth, "camera_lateral_right", obs_t)
                                #write_frame(left_video_rgb, left_video_depth, "camera_lateral_left", obs_t)
                                #write_frame(eye_in_hand_rgb, eye_in_hand_depth, "robot0_eye_in_hand", obs_t)
                                # get action
                                try:
                                    action_t = trajectory_obj[t]['action']
                                    logger.debug(f"Action at time-step {t}: {action_t}")
                                except KeyError:
                                    pass

                        cv2.destroyAllWindows()
