import os
import torch
from prettytable import PrettyTable
import yaml
import re
import pandas as pd

from constants import class_available, theme_available

save_base_path = "/home/rosca/TRDP-unlearning/UnlearnCanvaEval/assets/UC_test"

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

    result_table_s: dict[str, list] = {"Theme/Class": [], "UA": [], "IRA": [], "CRA": [], "FID": [], "Runtime": [], "Run_memory": []}

    folders = [f for f in os.listdir(save_base_path) if os.path.isdir(os.path.join(save_base_path, f))]
    print(f"Folders: {folders}")

    for forget_model in folders:

        path = os.path.join(save_base_path, forget_model)
        #print(f"Path: {path}")
        files = os.listdir(path)
        #print(f"Files: {files}")
        #print()
        if f"{forget_model}_style_eval.pth" in files and f"{forget_model}_class_eval.pth" and f"{forget_model}_fid_eval.pth" in files:
            print(f"Found data from {forget_model}")
            ua = 0.0
            ira = 0.0
            cra = 0.0
            fid = 0.0

            file_names = [f"{forget_model}_style_eval.pth", f"{forget_model}_class_eval.pth"]

            # evaluate in the same domain
            if forget_model in theme_available:
                file_indomain = f"{forget_model}_style_eval.pth"
                file_outdomain = f"{forget_model}_class_eval.pth"
                in_domain = theme_available
                out_domain = class_available

            else:
                file_indomain = f"{forget_model}_class_eval.pth"
                file_outdomain = f"{forget_model}_style_eval.pth"
                in_domain = class_available
                out_domain = theme_available

            data = torch.load(os.path.join(path, file_indomain), map_location=torch.device('cpu'))
            ua = data["acc"][forget_model]
            aux_acc = 0.0
            for other_theme in in_domain:
                if other_theme != forget_model:
                    aux_acc += data["acc"][other_theme]
            ira = aux_acc / (len(in_domain) - 1)

            # evaluate outside the domain
            data = torch.load(os.path.join(path, file_outdomain), map_location=torch.device('cpu'))
            aux_acc = 0.0
            for other_theme in out_domain:
                aux_acc += data["acc"][other_theme]
            cra = aux_acc / len(out_domain)

            # compute fid over all
            fid_file = f"{forget_model}_fid_eval.pth"
            fid = torch.load(os.path.join(path, fid_file), weights_only=False, map_location=torch.device('cpu'))

            # get memory allocation and time for unlearning
            model_data = load_model_card_yaml(os.path.join(path, "README.md"))

            runtime = retrieve_model_metric(model_data, "Runtime training seconds")
            memory = retrieve_model_metric(model_data, "Peak memory usage in training")

            result_table_s["Theme/Class"].append(forget_model)
            result_table_s["UA"].append(1.0 - to_float(ua))
            result_table_s["IRA"].append(to_float(ira))
            result_table_s["CRA"].append(to_float(cra))
            result_table_s["FID"].append(to_float(fid))
            result_table_s["Runtime"].append(float(runtime))
            result_table_s["Run_memory"].append(float(memory))

    print("\n\nModels evaluation:\n")
    pretty_table(result_table_s)

    print("\n\nAverages:\n")
    print(f"UA: {sum(result_table_s['UA'])/len(result_table_s['UA'])}")
    print(f"IRA: {sum(result_table_s['IRA'])/len(result_table_s['IRA'])}")
    print(f"CRA: {sum(result_table_s['CRA'])/len(result_table_s['CRA'])}")
    print(f"FID: {sum(result_table_s['FID'])/len(result_table_s['FID'])}")
    print(f"Runtime: {sum(result_table_s['Runtime'])/len(result_table_s['Runtime'])}")
    print(f"Run Memory: {sum(result_table_s['Run_memory'])/len(result_table_s['Run_memory'])}")

    # add averages and save tables
    result_table_s["Theme/Class"].append("Averages")
    result_table_s["UA"].append(sum(result_table_s["UA"])/len(result_table_s["UA"]))
    result_table_s["IRA"].append(sum(result_table_s["IRA"])/len(result_table_s["IRA"]))
    result_table_s["CRA"].append(sum(result_table_s["CRA"])/len(result_table_s["CRA"]))
    result_table_s["FID"].append(sum(result_table_s["FID"])/len(result_table_s["FID"]))
    result_table_s["Runtime"].append(sum(result_table_s["Runtime"])/len(result_table_s["Runtime"]))
    result_table_s["Run_memory"].append(sum(result_table_s["Run_memory"])/len(result_table_s["Run_memory"]))

    #style_res = pd.DataFrame(result_table_s)
    #style_res.to_csv(os.path.join(path, "results.csv"), index=False)