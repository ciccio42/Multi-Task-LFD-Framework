import csv
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import numpy as np
import json

JSON_FILE_PATH = "/home/frosa_loc/Multi-Task-LFD-Framework/utils/csv/task_results.json"
PLOT_NAME = "Task success rate per task"
X_LABEL = "Task name"
Y_LABEL = "Percentage"


def addlabels(x, y, std_dev):
    for i in range(len(x)):
        # "%.2f" % y[i])
        plt.text(i, y[i], f"{round(y[i], 2)} +- {round(std_dev[i], 2)}",
                 rotation=90,
                 rotation_mode='anchor',)


if __name__ == '__main__':

    # Opening JSON file
    with open(JSON_FILE_PATH) as f:
        data = json.load(f)
    print(data)

    success_rate_per_task = dict()
    for run_indx, run in enumerate(data.keys()):
        for task_indx, task_name in enumerate(data[run].keys()):
            if run_indx == 0:
                success_rate_per_task[task_name] = []

            success_rate_per_task[task_name].append(data[run][task_name])

    avg_success_rate_per_task = []
    std_success_rate_per_task = []
    task_name_list = []
    for task_name in success_rate_per_task.keys():
        task_name_list.append(task_name)
        avg_success_rate_per_task.append(
            np.mean(success_rate_per_task[task_name]))
        std_success_rate_per_task.append(
            np.std(success_rate_per_task[task_name]))

    width = 10
    height = 8
    figure = plt.figure(figsize=(width, height))
    print(task_name_list)
    plt.bar(np.array(task_name_list), np.array(avg_success_rate_per_task))
    addlabels(task_name_list, avg_success_rate_per_task,
              std_success_rate_per_task)
    plt.title(PLOT_NAME)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.errorbar(task_name_list, avg_success_rate_per_task, std_success_rate_per_task, fmt='.',
                 color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    plt.xticks(rotation=45)
    plt.savefig(f"{PLOT_NAME}.png")
