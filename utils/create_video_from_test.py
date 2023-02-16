import os
import pickle
import cv2
import re

# Set the directory to search for step-x folders
directory = "/user/frosa/robotic/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/2Task-NutAssembly-Batch45-1gpu-Attn2ly128-Act2ly64mix2-headCat-simclr128x256/results_nut_assembly"

# Define a custom sorting key function
def sort_key(file_name):
    # Extract the number X from the file name using a regular expression
    match = re.search(r"\d+", file_name)
    if match:
        return int(match.group())
    else:
        return 0  # Return 0 if the file name doesn't contain a number

# Find all the step-x folders
step_folders = [folder for folder in os.listdir(directory) if folder.startswith("step-")]

# Loop through each step folder
for step_folder in step_folders:
    print(step_folder)
    # Get the full path of the step folder
    step_folder_path = os.path.join(directory, step_folder)
    try:
        os.mkdir(os.path.join(step_folder_path, "video"))
    except:
        pass

    # Get a list of all the pickle files in the step folder
    demo_pkls = [file for file in os.listdir(step_folder_path) if file.endswith(".pkl") and ("demo" in file)]
    trj_pkls = [file for file in os.listdir(step_folder_path) if file.endswith(".pkl") and ("traj" in file)]
    demo_pkls.sort(key=sort_key)
    trj_pkls.sort(key=sort_key)

    for demo_file, trj_file in zip(demo_pkls, trj_pkls):
        # Load the first pickle file
        with open(os.path.join(step_folder_path, demo_file), 'rb') as f1:
            demo = pickle.load(f1)

        # Load the second pickle file
        with open(os.path.join(step_folder_path, trj_file), 'rb') as f2:
            trj = pickle.load(f2)

        # Check that both files have the same number of frames
        #if len(demo) != len(trj):
        #    raise ValueError('The two files must have the same number of frames')

        # Define the output video dimensions and FPS
        frame_width = trj[0]['obs']['image'].shape[1]
        frame_height = trj[0]['obs']['image'].shape[0]
        out_width = frame_width
        out_height = frame_height
        fps = 30

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        out = cv2.VideoWriter(os.path.join(step_folder_path, "video", f"{trj_file.split('.')[0]}.avi"), fourcc, fps, (out_width, out_height))

        # Loop through each frame and combine them side by side
        for i in range(len(trj)):
            # Get the frames from the two files
            #frame1 = trj[i]['obs']['image']
            frame2 = trj[i]['obs']['image']

            # Create the combined image
            out_frame = frame2 #cv2.hconcat([frame1, frame2])

            # Write the frame to the output video
            out.write(out_frame)

        # Release the video writer and close the files
        out.release()
        f1.close()
        f2.close()