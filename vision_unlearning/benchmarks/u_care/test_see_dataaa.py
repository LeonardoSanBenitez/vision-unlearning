import os
import torch
from prettytable import PrettyTable
import yaml
import re
import pandas as pd

from constants import class_available, theme_available


path = "assets/acc_debug/"

theme = "Picasso"

file_indomain = f"{theme}_style_eval.pth"
data = torch.load(os.path.join(path, file_indomain), map_location=torch.device('cpu'))
df = pd.DataFrame(data)
print(df.keys())
print(df.index)
for row in df.index:
    print(row, df.loc[row, "acc"])
# print(df)

if theme in theme_available:
    ua = data["acc"][theme]
    print(f"UA for {theme}: {1-ua}")


file_outdomain = f"{theme}_class_eval.pth"
data = torch.load(os.path.join(path, file_outdomain), map_location=torch.device('cpu'))
df = pd.DataFrame(data)
print(df.keys())
print(df.index)
for row in df.index:
    print(row, df.loc[row, "acc"])
# print(df)

if theme in class_available:
    ua = data["acc"][theme]
    print(f"UA for {theme}: {1-ua}")