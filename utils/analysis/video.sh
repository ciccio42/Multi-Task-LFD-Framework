#!/bin/bash

# Define the output video file name and codec
# Find all the "video" folders in the parent directory
parent_dir="/home/frosa_loc/Multi-Task-LFD-Framework/repo/TOSIL/one_shot_transformers/model/results/step-160000_1"
video_folders=($(find_video_folders "$parent_dir"))
output_file="$parent_dir/video.mp4"
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
        # video_files=("$video_folder"/*.mp4)
        echo $video_folder
        for video_file in $(find $video_folder/*.mp4 -type f | sort -V); do
            echo "file '$video_file'" >>"$parent_dir/output.txt"
        done
    done

    ffmpeg -f concat -safe 0 -i "$parent_dir/output.txt" -c copy "$output_file"
}

# Load and concatenate all the video files in the "video" folders
concatenate_videos "${video_folders[@]}"

echo "Concatenated ${#video_folders[@]} video files into '$output_file' ($(bc <<<"scale=2; $total_duration/1") seconds)."

rm "$parent_dir/output.txt"
