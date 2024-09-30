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
import matplotlib.pyplot as plt
from collections import OrderedDict
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat 
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

MAX_DIST = 10
MIN_DIST = 0.01

T_bl_sim_to_w_sim = np.array([[0, -1, 0, 0], 
                              [1, 0, 0, 0.612],
                              [0, 0, 1, -0.860],
                              [0, 0, 0, 1]])
T_world_to_bl = np.array([
                        [0, 1, 0, -0.612], 
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0.860],
                        [0, 0, 0, 1]])


R_g_sim_to_g_robot = np.array([[0, -1, 0], 
                              [1, 0, 0],
                              [0, 0, 1]])
# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

# x, y, z, e_x, e_y, e_z
# NORM_RANGES = np.array([[-0.35,  0.35],
#                         [-0.35,  0.35],
#                         [0.60,  1.20],
#                         [-3.14,  3.14911766],
#                         [-3.14911766, 3.14911766],
#                         [-3.14911766,  3.14911766]])
NORM_RANGES = np.array([[-0.40,  0.40],
                        [0.10,  0.90],
                        [-0.20,  0.20],
                        [-3.14911766,  3.14911766],
                        [-3.14911766, 3.14911766],
                        [-3.14911766,  3.14911766]])
ACTION_DISTRIBUTION = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

TABLE_SIZE = [100, 100]  # cmxcm

max_len_trj = 0

# Function to extract the numerical part from the file name
def extract_number(file_path):
    match = re.search(r'traj(\d+)\.pkl', file_path)
    if match:
        return int(match.group(1))
    return -1  # In case the file name doesn't match the pattern


def trasform_from_world_to_bl(action):
    aa_gripper = action[3:-1]
    # convert axes-angle into rotation matrix
    R_w_sim_to_gripper_sim = quat2mat(axisangle2quat(aa_gripper))
    
    gripper_pos = action[0:3]
    
    T_w_sim_gripper_sim = np.zeros((4,4))
    T_w_sim_gripper_sim[3,3] = 1
    
    # position
    T_w_sim_gripper_sim[0,3] = gripper_pos[0]
    T_w_sim_gripper_sim[1,3] = gripper_pos[1]
    T_w_sim_gripper_sim[2,3] = gripper_pos[2]
    # orientation
    T_w_sim_gripper_sim[0:3, 0:3] = R_w_sim_to_gripper_sim
    
    T_bl_sim_gripper_sim = T_bl_sim_to_w_sim @ T_w_sim_gripper_sim
    
    # print(f"Transformation from world to bl:\n{T_bl_sim_gripper_sim}")
    
    R_bl_to_gripper_sim = T_bl_sim_gripper_sim[0:3, 0:3]
    
    R_bl_to_gripper_real = R_bl_to_gripper_sim @ R_g_sim_to_g_robot
    
    action_bl = np.zeros((7))
    action_bl[0:3] = T_bl_sim_gripper_sim[0:3, 3]
    action_bl[3:6] = quat2axisangle(mat2quat(R_bl_to_gripper_real))
    action_bl[6] = action[-1]
    
    return action_bl

def trasform_bl_to_world(action):
    aa_gripper = action[3:-1]
    # convert axes-angle into rotation matrix
    R_bl_to_gripper = quat2mat(axisangle2quat(aa_gripper))
    
    gripper_pos = action[0:3]
    
    T_bl_to_gripper = np.zeros((4,4))
    T_bl_to_gripper[3,3] = 1
    
    # position
    T_bl_to_gripper[0,3] = gripper_pos[0]
    T_bl_to_gripper[1,3] = gripper_pos[1]
    T_bl_to_gripper[2,3] = gripper_pos[2]
    # orientation
    T_bl_to_gripper[0:3, 0:3] = R_bl_to_gripper
    
    T_world_to_gripper = T_world_to_bl @ T_bl_to_gripper
    
    # print(f"Transformation from world to bl:\n{T_bl_sim_gripper_sim}")
    
    R_world_to_gripper = T_world_to_gripper[0:3, 0:3]

    
    action_world = np.zeros((7))
    action_world[0:3] = T_world_to_gripper[0:3, 3]
    action_world[3:6] = quat2axisangle(mat2quat(R_world_to_gripper))
    action_world[6] = action[-1]
    
    return action_world


def heat_map(task_distribution, task_path, task_name):
    # Each px represents 0.5x0.5 cm
    px_resolution = 0.5

    # Compute table size in pixels
    table_size_px = np.array(
        np.array(TABLE_SIZE) / px_resolution, dtype=np.int32)

    # Initialize the table map (heatmap matrix) with zeros
    table_map = np.zeros((table_size_px[0], table_size_px[1]))

    # Loop through task variations and their respective trajectories
    for variation in task_distribution.keys():
        print(f"Processing variation: {variation}")

        for trj in task_distribution[variation].keys():
            # Convert the x, y positions from meters to pixel resolution (action_t[0] -> vertical, action_t[1] -> horizontal)
            trajectory = np.array(
                [(action_t[:2] * 100) / px_resolution for action_t in task_distribution[variation][trj]],
                dtype=np.int32
            )

            # Adjust the coordinates for heatmap plotting
            # Positive x-values (action_t[0]) go down, so flip the sign for positive values
            positive_flags_x = trajectory[:, 0] > 0
            negative_flags_x = trajectory[:, 0] <= 0
            positive_flags_y = trajectory[:, 1] > 0
            negative_flags_y = trajectory[:, 1] <= 0

            # Map x-coordinates (action_t[0] -> vertical axis)
            trajectory[:, 0][positive_flags_x] = int(table_map.shape[0] / 2) + trajectory[:, 0][positive_flags_x]
            trajectory[:, 0][negative_flags_x] = int(table_map.shape[0] / 2) - (-trajectory[:, 0][negative_flags_x])

            # Map y-coordinates (action_t[1] -> horizontal axis)
            trajectory[:, 1][positive_flags_y] = int(table_map.shape[1] / 2) + trajectory[:, 1][positive_flags_y]
            trajectory[:, 1][negative_flags_y] = int(table_map.shape[1] / 2) - (-trajectory[:, 1][negative_flags_y])

            # Update the heatmap with the trajectory points
            for x, y in trajectory:
                table_map[x, y] += 1  # Increment the density at that (x, y) point

    if 'pick_place' in task_name:
        y_min, y_max = -30, 30
        x_min, x_max = -35, 35
        task = "Pick-Place"
    elif 'nut_assembly' in task_name:
        y_min, y_max = -30, 30
        x_min, x_max = -35, 35
        task = "Nut-Assembly"
    elif 'stack_block' in task_name:
        y_min, y_max = -30, 30
        x_min, x_max = -35, 35
        task = "Stack-Block"
    else:
        y_min, y_max = -30, 30
        x_min, x_max = -35, 35
        task = "Press-Button"


    # Convert to pixel indices based on px_resolution
    y_min_px = int((y_min + TABLE_SIZE[0] / 2) / px_resolution)
    y_max_px = int((y_max + TABLE_SIZE[0] / 2) / px_resolution)
    x_min_px = int((x_min + TABLE_SIZE[1] / 2) / px_resolution)
    x_max_px = int((x_max + TABLE_SIZE[1] / 2) / px_resolution)

    # Crop the table_map matrix
    cropped_table_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # Plot the cropped heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(f"{task}: heatmap of x-y plane trajectories")

    # Set axis labels to centimeters
    plt.xlabel("Y table axis (cm)")
    plt.ylabel("X table axis (cm)")

    # Use logarithmic normalization to make low values more visible
    norm = mcolors.LogNorm(vmin=1, vmax=np.max(cropped_table_map))

    # Display the cropped heatmap with adjusted color scale and flip the y-axis
    im = ax.imshow(cropped_table_map, cmap='plasma', origin='upper', norm=norm)

    # Adjust the ticks to reflect the cropped region in centimeters
    ticks_x = np.arange(0, cropped_table_map.shape[1], step=int(10 / px_resolution))  # Every 10 cm
    ticks_y = np.arange(0, cropped_table_map.shape[0], step=int(10 / px_resolution))  # Every 10 cm

    # Set the tick labels corresponding to the cropped region
    tick_labels_x = np.arange(x_min, x_max, step=10)  # Corresponding labels in cm
    tick_labels_y = np.arange(y_min, y_max, step=10)  # Corresponding labels in cm

    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    # Create a colorbar axis and adjust its size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add the color bar with adjusted height
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Number of Trajectories (log scale)")

    # Save the heatmap to the specified file path
    plt.savefig(os.path.join(task_path, f"{task_name}_heatmap_limited.png"))

    # Optionally, show the plot (useful for interactive environments)
    # plt.show()


def normalize_action(action, n_action_bin, action_ranges, continous=False):
    half_action_bin = int(n_action_bin/2)
    norm_action = action.copy()
    # normalize between [-1 , 1]
    if action.shape[0] == 7:
        norm_action[:-1] = (2 * (norm_action[:-1] - action_ranges[:, 0]) /
                            (action_ranges[:, 1] - action_ranges[:, 0])) - 1

    else:
        norm_action = (2 * (norm_action - action_ranges[:, 0]) /
                       (action_ranges[:, 1] - action_ranges[:, 0])) - 1
    if continous:
        return norm_action
    else:
        # action discretization
        return (norm_action * half_action_bin).astype(np.int32).astype(np.float32) / half_action_bin


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument("--debug", action='store_true',
                        help="whether or not attach the debugger")
    parser.add_argument("--depth", action='store_true',
                        help="whether or not render depth")
    parser.add_argument("--min_max", action='store_true',
                        help="whether compute min_max")
    parser.add_argument("--real", action='store_true',
                        help="whether the dataset is real")
    args = parser.parse_args()

    camera_name = "image"

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()

    logger.info(f"Task path: {args.task_path}")
    i = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    task_name = args.task_path.split('/')[-1]
    task_paths = glob.glob(os.path.join(args.task_path, 'task_*'))
    
    task_paths_limited = []
    for task_path in task_paths:
        if 'task_00' in task_path or 'task_01' in task_path or 'task_04' in task_path or'task_05' in task_path or 'task_08' in task_path or 'task_09' in task_path:
            task_paths_limited.append(task_path)
    
    task_paths = task_paths_limited
    global_trajectory_lengths = [] 
    
    if not args.min_max:
        task_distribution = OrderedDict()
        for task_var, dir in enumerate(sorted(task_paths)):

            task_distribution[task_var] = OrderedDict()
            print(dir)

            if os.path.isdir(os.path.join(args.task_path, dir)):
                # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
                trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))
                norm_action = []
                action = []
                trj_paths = sorted(trj_paths, key=extract_number)
                trj_paths = trj_paths[:40]
                for j, trj in enumerate(sorted(trj_paths)):

                    task_distribution[task_var][j] = list()
                    print(trj)
                    
                    with open(trj, "rb") as f:
                        sample = pickle.load(f)
                        logger.debug(f"Sample keys {sample.keys()}")
                        logger.debug(sample)
                        # print(f"Sample command: {sample['command']}")
                        if i == 0:
                            i += 1
                            logger.debug(sample)
                            logger.debug(
                                f"Observation keys: {sample['traj'][0]['obs'].keys()}")
                            logger.debug(
                                f"{sample['traj'][0]['obs']['ee_aa']}")

                        # take the Trajectory obj from the trajectory
                        trajectory_obj = sample['traj']
                        global_trajectory_lengths.append(len(trajectory_obj))
                        i = 0
                        obj_in_hand = 0
                        start_moving = 0
                        end_moving = 0
                        for t in range(len(trajectory_obj)):
                            if t > 0:
                                logger.debug(f"Time-step {t}")
                                try:
                                    action_t = trajectory_obj.get(t)[
                                        'action']
                                    
                                    # task_distribution[task_var][j].append(
                                    #     normalize_action(action=action_t,
                                    #                      n_action_bin=256,
                                    #                      action_ranges=NORM_RANGES))
                                    if  args.real:
                                        rot_quat = quat2axisangle(action_t[3:7])
                                        action_list = list()
                                        action_list.extend(action_t[:3])
                                        action_list.extend(rot_quat)
                                        action_list.extend([action_t[-1]])
                                        action_t = np.array(action_list)
                                        norm_action.append(normalize_action(action=action_t,
                                                                            n_action_bin=256,
                                                                            action_ranges=NORM_RANGES))
                                        action.append(action_t)
                                    else:
                                        action.append(action_t)
                                        norm_action.append(normalize_action(action=action_t,
                                                                            n_action_bin=256,
                                                                            action_ranges=NORM_RANGES))
                                    
                                    if args.real:
                                        task_distribution[task_var][j].append(trasform_bl_to_world(action_t))
                                    else:
                                        task_distribution[task_var][j].append(action_t)
                                        # transform the action from the world to the baselink
                                        # action_bl = trasform_from_world_to_bl(action)
                                    # for dim, action_label in enumerate(action_t):
                                    #     ACTION_DISTRIBUTION[dim].append(
                                    #         action_label)
                                    # print(f"Norm action{action_t[:3]}")
                                    # print(
                                    #     f"Denorm action {denormalize_action(action_t)[:3]}")
                                except KeyError:
                                    print("error")
        # Compute mean and std-deviation
        norm_action_matrix = np.array(norm_action).reshape(len(norm_action), 7)
        action_matrix = np.array(action).reshape(len(action), 7)
        print(
            f"Standard deviation {np.std(np.array(norm_action_matrix), axis=0)}")
        print(f"Standard mean {np.mean(np.array(norm_action_matrix), axis=0)}")


        heat_map(task_distribution=task_distribution,
                 task_name=task_name,
                 task_path=args.task_path)
        
        mean_trajectory_len = np.mean(global_trajectory_lengths)
        std_trajectory_len = np.std(global_trajectory_lengths)
        print(f"Mean trajectory lenght {mean_trajectory_len} - Std trajectory len {std_trajectory_len}")
        
        for variation in task_distribution.keys():
            print(variation)
            # Plot y-axis trajectories for each variation
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(
                16, 9))
            fig.tight_layout(pad=5.0)
            for trj in task_distribution[variation].keys():
                trajectory = np.array(
                    [action_t[:3] for action_t in task_distribution[variation][trj]])

                t = np.array([i for i in range(len(trajectory))])
                ax1.plot(t, trajectory[:, 0], color='b', alpha=0.5)
                ax1.set_title(f"Trajectory distribution along x axis")
                ax1.set_xlabel("Timestamp t")
                ax1.set_ylabel("scaled x value")

                ax2.plot(t, trajectory[:, 1], color='b', alpha=0.5)
                ax2.set_title(f"Trajectory distribution along y axis")
                ax2.set_xlabel("Timestamp t")
                ax2.set_ylabel("scaled y value")

                ax3.plot(t, trajectory[:, 2], color='b', alpha=0.5)
                ax3.set_title(f"Trajectory distribution along z axis")
                ax3.set_xlabel("Timestamp t")
                ax3.set_ylabel("scaled z value")

            plt.savefig(os.path.join(args.task_path, f"{task_name}_variation_{variation}.png"))

        # An "interface" to matplotlib.axes.Axes.hist() method
        for dim, key in enumerate(ACTION_DISTRIBUTION.keys()):
            n, bins, patches = plt.hist(x=ACTION_DISTRIBUTION[key], bins=256, color='#0504aa',
                                        alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Action Distribution')
            maxfreq = n.max()
            # Set a clean upper y-axis limit.
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq %
                     10 else maxfreq + 10)
            plt.savefig(os.path.joint(args.task_path, f"dim_{dim}.png"))
            plt.clf()
    else:
        for task_var, dir in enumerate(sorted(task_paths)):
            if task_var == 0:
                if os.path.isdir(dir):  # os.path.join(args.task_path, dir)):
                    # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
                    trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))
                    norm_action = []
                    for j, trj in enumerate(sorted(trj_paths)):

                        print(trj)

                        with open(trj, "rb") as f:
                            sample = pickle.load(f)
                            logger.debug(f"Sample keys {sample.keys()}")
                            logger.debug(sample)
                            # print(f"Sample command: {sample['command']}")
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
                            if len(trajectory_obj) > max_len_trj:
                                max_len_trj = len(trajectory_obj)
                            for t in range(len(trajectory_obj)):
                                if t > 0:
                                    logger.debug(f"Time-step {t}")
                                    # try:
                                    action_t = trajectory_obj.get(t)[
                                        'action']
                                    
                                    if  args.real:
                                        rot_quat = quat2axisangle(action_t[3:7])
                                        action_list = list()
                                        action_list.extend(action_t[:3])
                                        action_list.extend(rot_quat)
                                        action_list.extend([action_t[-1]])
                                        action_t = np.array(action_list)
                                        norm_action.append(normalize_action(action=action_t,
                                                                            n_action_bin=256,
                                                                            action_ranges=NORM_RANGES))
                                    else:
                                        # transform the action from the world to the baselink
                                        img = trajectory_obj.get(t)['obs']['camera_front_image'][:,:,::-1]
                                        cv2.imwrite("img.png", img)
                                        action_bl = trasform_from_world_to_bl(action_t)
                                        action_t = action_bl
                                    # next_action_t = trajectory_obj.get(t+5)[
                                    #     'action']
                                    # print(
                                    #     f"Position {action_t[:3] - next_action_t[:3]}\n{next_action_t[7]}")
                                    # rot_quat = quat2axisangle(action_t[3:7])
                                    # action = np.concatenate(
                                    #     (action_t[:3], rot_quat, [action_t[7]]))
                                    try:
                                        for dim, action_label in enumerate(action_t):
                                            ACTION_DISTRIBUTION[dim].append(
                                                action_label)
                                    except KeyError:
                                        print("error")

        for indx, action_dim in enumerate(ACTION_DISTRIBUTION.values()):
            print(f"Dim {indx} - Min {min(action_dim)} - Max {max(action_dim)}")

        print(f"Max len trj {max_len_trj}")
