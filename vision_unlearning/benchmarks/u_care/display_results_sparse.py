import os
import torch
from prettytable import PrettyTable
import yaml
import re
import pandas as pd

from constants import class_available, theme_available

objects=[
    "Architectures",
    "Bears",
    "Birds",
    "Butterfly",
    "Cats",
    "Dogs",
    "Fishes",
    "Flame",
    "Flowers",
    "Frogs"
]

styles = [
    "Artist_Sketch",
    "Bricks",
    "Byzantine",
    "Cartoon",
    "Cold_Warm",
    "Color_Fantasy",
    "Comic_Etch",
    "Crayon",
    "Cubism",
    "Dadaism"
]

base_path = "/home/rosca/TRDP-unlearning/UnlearnCanvaEval/assets/sparse/mini_uc_results"
n_images = 50

def pretty_table(dct):
    table = PrettyTable()
    for c in dct.keys():
        table.add_column(c, [])
    table.add_row(['\n'.join(map(str, dct[c])) for c in dct.keys()])
    print(table)


def to_float(x):
    return x.item() if hasattr(x, "item") else x

def load_model_card_yaml(path):
    with open(path, "r") as f:
        text = f.read()

    # Extract YAML front-matter (between --- markers)
    match = re.search(r"---\n(.*?)\n---", text, re.S)
    if not match:
        raise ValueError("No YAML front-matter found")

    return yaml.safe_load(match.group(1))

def retrieve_model_metric(model_data, metric_name):
    for entry in model_data["model-index"]:
        for result in entry.get("results", []):
            for metric in result.get("metrics", []):
                name = metric.get("name", "")
                if name.startswith(metric_name):
                    return metric["value"]
    return None


if __name__ == "__main__":

    avg_acc_style = 0.0
    avg_acc_class = 0.0
    avg_ira_style = 0.0
    avg_ira_class = 0.0
    avg_cra_style = 0.0
    avg_cra_class = 0.0

    result_table_s: dict[str, list] = {"Theme/Class": [], "UA": [], "IRA": [], "CRA": []}

    concepts = objects + styles
    print(f"concepts: {concepts}")
    files = os.listdir(base_path)

    for forget_model in concepts:

        if f"{forget_model}_style_ckpt500.pth" in files and f"{forget_model}_class_ckpt500.pth" in files:
            print(f"Found data from {forget_model}")
            ua = 0.0
            ira = 0.0
            cra = 0.0

            file_names = [f"{forget_model}_style_ckpt500.pth", f"{forget_model}_class_ckpt500.pth"]

            # evaluate in the same domain
            if forget_model in theme_available:
                file_indomain = f"{forget_model}_style_ckpt500.pth"
                file_outdomain = f"{forget_model}_class_ckpt500.pth"
                in_domain = styles
                out_domain = objects

            else:
                file_indomain = f"{forget_model}_class_ckpt500.pth"
                file_outdomain = f"{forget_model}_style_ckpt500.pth"
                in_domain = objects
                out_domain = styles

            data = torch.load(os.path.join(base_path, file_indomain), map_location=torch.device('cpu'))
            ua = data["acc"][forget_model]/n_images
            aux_acc = 0.0
            for other_theme in in_domain:
                if other_theme != forget_model:
                    aux_acc += data["acc"][other_theme]/n_images
            ira = aux_acc / (len(in_domain) - 1)

            # evaluate outside the domain
            data = torch.load(os.path.join(base_path, file_outdomain), map_location=torch.device('cpu'))
            aux_acc = 0.0
            for other_theme in out_domain:
                aux_acc += data["acc"][other_theme]/n_images
            cra = aux_acc / len(out_domain)

            result_table_s["Theme/Class"].append(forget_model)
            result_table_s["UA"].append(1.0 - to_float(ua))
            result_table_s["IRA"].append(to_float(ira))
            result_table_s["CRA"].append(to_float(cra))

            if forget_model in styles:
                avg_acc_style += 1.0 - to_float(ua)
                avg_ira_style += to_float(ira)
                avg_cra_style += to_float(cra)
            else:
                avg_acc_class += 1.0 - to_float(ua)
                avg_ira_class += to_float(ira)
                avg_cra_class += to_float(cra)

    print("\n\nModels evaluation:\n")
    pretty_table(result_table_s)

    print("\n\nAverages:\n")
    print(f"UA: {sum(result_table_s['UA'])/len(result_table_s['UA'])}")
    print(f"IRA: {sum(result_table_s['IRA'])/len(result_table_s['IRA'])}")
    print(f"CRA: {sum(result_table_s['CRA'])/len(result_table_s['CRA'])}")

    # add averages and save tables
    result_table_s["Theme/Class"].append("Averages")
    result_table_s["UA"].append(sum(result_table_s["UA"])/len(result_table_s["UA"]))
    result_table_s["IRA"].append(sum(result_table_s["IRA"])/len(result_table_s["IRA"]))
    result_table_s["CRA"].append(sum(result_table_s["CRA"])/len(result_table_s["CRA"]))

    print(f"Average Classes:")
    print(f"UA: {((avg_acc_class/len(objects))*100):.2f} IRA: {((avg_ira_class/len(objects))*100):.2f} CRA: {((avg_cra_class/len(objects))*100):.2f}")

    print(f"Average Styles:")
    print(f"UA: {((avg_acc_style/len(styles))*100):.2f} IRA: {((avg_ira_style/len(styles))*100):.2f} CRA: {((avg_cra_style/len(styles))*100):.2f}")

    #style_res = pd.DataFrame(result_table_s)
    #style_res.to_csv(os.path.join(path, "results.csv"), index=False)