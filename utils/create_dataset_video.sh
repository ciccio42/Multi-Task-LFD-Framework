#!/bin/bash

# Set the parent directory
parent_dir="/mnt/sdc1/frosa/multitask_dataset/multitask_dataset_baseline/stack_block/panda_stack_block"

# Find the video directory
video_dir="$parent_dir/video"

# Find all task subdirectories
task_dirs=($(find "$video_dir" -mindepth 1 -maxdepth 1 -type d -name 'task_*'))

# Loop over each task subdirectory
for dir in "${task_dirs[@]}"; do
  # Get the task number from the directory name
  task_num=$(basename "$dir" | sed 's/task_//')
  
  # Find a random trajectory subdirectory containing the .avi video
  traj_dir=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'traj*' | shuf -n 1)
  video_file=$(find "$traj_dir" -type f -name '*.avi' | head -n 1)
  # Append the video file to the output file
  echo "file '$(realpath "$video_file")'" >> "$parent_dir/output.txt"
done

# Concatenate the videos in ascending order of task number
sort -t _ -k 2n "$parent_dir/output.txt" -o "$parent_dir/output.txt"

ffmpeg -f concat -safe 0 -i "$parent_dir/output.txt" -c:v libx264 -b:v 1000k -c copy "$parent_dir/output.avi"

rm $parent_dir/output.txt
