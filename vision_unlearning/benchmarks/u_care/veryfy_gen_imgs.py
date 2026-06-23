import os

root_folder = "assets/gen_img_samples_best/"
n_files = 5100

# get list of folders
folders = os.listdir(root_folder)

for folder in folders:
    path = os.path.join(root_folder, folder)
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) != n_files:
            print(f"Did not generate all images for {folder}")