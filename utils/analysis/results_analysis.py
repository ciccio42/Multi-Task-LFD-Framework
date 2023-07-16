import os, glob, json

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="/")
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--step', type=int, default=-1)
    parser.add_argument('--task_indx', type=int, default=-1)
    args = parser.parse_args()     

    # 1. Create path to desired results
    if args.step != -1:
        results_path = os.path.join(args.results_dir, f"results_{args.task_name}", f"step-{args.step}")
    elif args.task_indx != -1 and  args.step == -1:
        results_path = os.path.join(args.results_dir, f"results_{args.task_name}", f"task-{args.task_indx}")
    else:
        raise Exception("Specify either step or task-id")
    
    desired_variation_results = args.task_indx

    # 2. Open results file
    picked_counter = 0
    reached_counter = 0
    success_counter = 0
    for file in glob.glob(os.path.join(results_path, "traj*.json")):
        # open json file
        with open(file, "rb") as f:
            traj_result = json.load(f)
        if traj_result['variation_id'] == desired_variation_results:
            picked_counter = picked_counter + 1 if traj_result["picked"] else picked_counter
            reached_counter = reached_counter + 1 if traj_result["reached"] else reached_counter
            success_counter = success_counter + 1 if traj_result["success"] else success_counter

    print(f"Reached {reached_counter} - Picked {picked_counter} - Success {success_counter}")


