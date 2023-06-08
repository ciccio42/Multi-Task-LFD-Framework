import csv
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

CVS_FILE_PATH = "/home/frosa_loc/Multi-Task-LFD-Framework/utils/csv/wandb_export_2023-06-07T09_18_38.580+02_00.csv"
PLOT_NAME = "avg_success"
X_LABEL = "Step"
Y_LABEL = "Percentage"

# function to add value labels


def addlabels(x, y, std_dev):
    for i in range(len(x)):
        # "%.2f" % y[i])
        plt.text(i, y[i], f"{round(y[i], 2)} +- {round(std_dev[i], 2)}")


if __name__ == '__main__':

    # Open CSV file
    results_dict = OrderedDict()
    with open(CVS_FILE_PATH, newline='') as csvfile:
        results = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, row in enumerate(results):
            if i > 0:
                experiment_name = row[0].split("Name: ")[1]
                experiment_results = float(row[2])
                if experiment_name not in results_dict:
                    results_dict[experiment_name] = []
                results_dict[experiment_name].append(experiment_results)

    results_label = []
    results_avg = []
    results_std_dev = []
    for key, value in results_dict.items():
        results_label.append(key.split("-")[-1].split("_")[-1])
        results_avg.append(np.mean(value))
        results_std_dev.append(np.std(value))

    sorting_indices = np.argsort(results_label)
    results_label = np.array(results_label)[sorting_indices]
    results_avg = np.array(results_avg)[sorting_indices]
    results_std_dev = np.array(results_std_dev)[sorting_indices]

    width = 10
    height = 8
    figure = plt.figure(figsize=(width, height))
    plt.bar(results_label, results_avg)
    addlabels(results_label, results_avg, results_std_dev)
    plt.title(PLOT_NAME)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.errorbar(results_label, results_avg, results_std_dev, fmt='.', color='Black', elinewidth=2,
                 capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    plt.savefig(f"{PLOT_NAME}.png")
