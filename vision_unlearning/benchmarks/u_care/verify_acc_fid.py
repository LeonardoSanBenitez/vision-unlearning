import os
from constants import class_available, theme_available

root_folder = "assets/accuracy_results/"

all_themes = class_available + theme_available

# get list of folders
files = os.listdir(root_folder)

# check accuracy
for theme in all_themes:
    if theme == "Seed_Images":
        continue
    if f"{theme}_class_eval.pth" not in files:
        print(f"Accuracy: Class eval not run for {theme}!")
    if f"{theme}_style_eval.pth" not in files:
        print(f"Accuracy: Style eval not run for {theme}!")

# check fid
for theme in all_themes:
    if theme == "Seed_Images":
        continue
    if f"{theme}_fid.pth" not in files:
        print(f"FID eval not run for {theme}!")