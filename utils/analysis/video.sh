#!/bin/bash

# Define the output video file name and codec
output_file="output.mp4"
fourcc="mp4v"
# Define the function to recursively search for "video" folders
find_video_folders() {
    local parent_dir="$1"
    local video_folders=()
    while IFS= read -r -d '' dir; do
        video_folders+=("$dir")
    done < <(find "$parent_dir" -type d -name "video" -print0)
    echo "${video_folders[@]}"
}

# Define the function to load and concatenate the video files
concatenate_videos() {
    local video_folders=("$@")
    local video_files=()
    local total_duration=0
    for video_folder in "${video_folders[@]}"; do
        video_files=("$video_folder"/*.mp4)
        for video_file in "${video_files[@]}"; do
            echo "file '$video_file'" >> videos.txt;
        done
    done
    echo ${video_files[@]}
    ffmpeg -f concat -safe 0 -i videos.txt -c copy "$output_file"
}

# Find all the "video" folders in the parent directory
parent_dir="/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline/Task-NutAssembly-Batch27-1gpu-Attn2ly128-Act2ly256mix4-headCat-simclr128x512/results_nut_assembly/"
video_folders=($(find_video_folders "$parent_dir"))

# Load and concatenate all the video files in the "video" folders
concatenate_videos "${video_folders[@]}"

echo "Concatenated ${#video_folders[@]} video files into '$output_file' ($(bc <<< "scale=2; $total_duration/1") seconds)."

rm videos.txt