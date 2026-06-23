import os
import re
import pandas as pd
from constants import class_available
import numpy as np

forget_folder = "assets/models_debug"

def extract_stats(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    time_seconds = None
    peak_gpu_gb = None

    time_match = re.search(r"Total training time \(seconds\):\s*([\d.]+)", text)
    gpu_match = re.search(r"Peak GPU memory allocated:\s*([\d.]+)\s*GB", text)

    if time_match:
        time_seconds = float(time_match.group(1))

    if gpu_match:
        peak_gpu_gb = float(gpu_match.group(1))

    return time_seconds, peak_gpu_gb


def extract_stats_old(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    time_seconds = None
    peak_gpu_gb = None

    time_match = re.search(
        r"Runtime training \(s\):\s*([\d.]+)", text
    )
    gpu_match = re.search(
        r"Peak GPU memory usage \(GB\):\s*([\d.]+)", text
    )

    if time_match:
        time_seconds = float(time_match.group(1))

    if gpu_match:
        peak_gpu_gb = float(gpu_match.group(1))

    return time_seconds, peak_gpu_gb


folders = os.listdir(forget_folder)
results = {"Folder": [], "training_time": [], "training_mem": []}

for folder in folders:
    file = os.path.join(forget_folder, folder, "model_stats.txt")
    if os.path.exists(file):
        time_seconds, peak_gpu_gb = extract_stats_old(file)
        if peak_gpu_gb > 13:
            results["Folder"].append(folder)
            results["training_time"].append(time_seconds)
            results["training_mem"].append(peak_gpu_gb)

df = pd.DataFrame(results)
df.to_csv("time_memory_results_old.csv", index=None)

obj_time = []
obj_mem = []
style_time = []
style_mem = []
# print mean values
for i in range(len(results["Folder"])):
    if results["Folder"][i].split('_')[0] in class_available:
        obj_time.append(results["training_time"][i])
        obj_mem.append(results["training_mem"][i])
    else:
        style_time.append(results["training_time"][i])
        style_mem.append(results["training_mem"][i])

print("Displaying Means")
print()
print("Objects metrics")
print(f"Mean training time (s): {np.mean(obj_time):.2f}+-{np.std(obj_time):.2f}")
print(f"Mean training memory (GB): {np.mean(obj_mem):.2f}+-{np.std(obj_mem):.2f}")
print()
print("Styles metrics")
print(f"Mean training time (s): {np.mean(style_time):.2f}+-{np.std(style_time):.2f}")
print(f"Mean training memory (GB): {np.mean(style_mem):.2f}+-{np.std(style_mem):.2f}")
