import os
import torch
from prettytable import PrettyTable
import yaml
import re
import pandas as pd

from constants import class_available, theme_available


path = "assets/accuracy_results/"
path_models = "assets/models/"

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

    files = os.listdir(path)

    all_themes = theme_available + class_available

    for theme in theme_available:
        ua = 0.0
        ira = 0.0
        cra = 0.0
        fid = 0.0

        if f"{theme}_style_eval.pth" in files and f"{theme}_class_eval.pth" and f"{theme}_fid.pth" in files:
            file_names = [f"{theme}_style_eval.pth", f"{theme}_class_eval.pth"]

            # evaluate in the same domain
            if theme in theme_available:
                file_indomain = f"{theme}_style_eval.pth"
                file_outdomain = f"{theme}_class_eval.pth"
                in_domain = theme_available
                out_domain = class_available

            else:
                file_indomain = f"{theme}_class_eval.pth"
                file_outdomain = f"{theme}_style_eval.pth"
                in_domain = class_available
                out_domain = theme_available

            data = torch.load(os.path.join(path, file_indomain), map_location=torch.device('cpu'))
            ua = data["acc"][theme]
            aux_acc = 0.0
            for other_theme in in_domain:
                if other_theme != theme:
                    aux_acc += data["acc"][other_theme]
            ira = aux_acc / (len(in_domain) - 1)

            # evaluate outside the domain
            data = torch.load(os.path.join(path, file_outdomain), map_location=torch.device('cpu'))
            aux_acc = 0.0
            for other_theme in out_domain:
                aux_acc += data["acc"][other_theme]
            cra = aux_acc / len(out_domain)

            # compute fid over all
            fid_file = f"{theme}_fid.pth"
            fid = torch.load(os.path.join(path, fid_file), weights_only=False, map_location=torch.device('cpu'))

            # get memory allocation and time for unlearning
            distill_folder = os.listdir(os.path.join(path_models, f"forget_style_{theme}"))
            if len(distill_folder) > 0:
                distill_f = distill_folder[0]
            else:
                distill_f = distill_folder[0]

            model_data = load_model_card_yaml(os.path.join(path_models, f"forget_style_{theme}", distill_f, "README.md"))

            runtime = retrieve_model_metric(model_data, "Runtime training seconds")
            memory = retrieve_model_metric(model_data, "Peak memory usage in training")

            result_table_s["Theme/Class"].append(theme)
            result_table_s["UA"].append(1.0 - to_float(ua))
            result_table_s["IRA"].append(to_float(ira))
            result_table_s["CRA"].append(to_float(cra))
            result_table_s["FID"].append(to_float(fid))
            result_table_s["Runtime"].append(float(runtime))
            result_table_s["Run_memory"].append(float(memory))

    print("\n\nStyles evaluation:\n")
    pretty_table(result_table_s)

    print("\n\nAverages Styles:\n")
    print(f"UA: {sum(result_table_s['UA'])/len(result_table_s['UA'])}")
    print(f"IRA: {sum(result_table_s['IRA'])/len(result_table_s['IRA'])}")
    print(f"CRA: {sum(result_table_s['CRA'])/len(result_table_s['CRA'])}")
    print(f"FID: {sum(result_table_s['FID'])/len(result_table_s['FID'])}")
    print(f"Runtime: {sum(result_table_s['Runtime'])/len(result_table_s['Runtime'])}")
    print(f"Run Memory: {sum(result_table_s['Run_memory'])/len(result_table_s['Run_memory'])}")

    result_table_c: dict[str, list] = {"Theme/Class": [], "UA": [], "IRA": [], "CRA": [], "FID": [], "Runtime": [], "Run_memory": []}
    for theme in class_available:
        ua = 0.0
        ira = 0.0
        cra = 0.0
        fid = 0.0

        if f"{theme}_style_eval.pth" in files and f"{theme}_class_eval.pth" and f"{theme}_fid.pth" in files:
            file_names = [f"{theme}_style_eval.pth", f"{theme}_class_eval.pth"]

            # evaluate in the same domain
            if theme in class_available:
                file_indomain = f"{theme}_class_eval.pth"
                file_outdomain = f"{theme}_style_eval.pth"
                in_domain = class_available
                out_domain = theme_available

            else:
                file_indomain = f"{theme}_style_eval.pth"
                file_outdomain = f"{theme}_class_eval.pth"
                in_domain = theme_available
                out_domain = class_available

            data = torch.load(os.path.join(path, file_indomain), map_location=torch.device('cpu'))
            ua = data["acc"][theme]
            aux_acc = 0.0
            for other_theme in in_domain:
                if other_theme != theme:
                    aux_acc += data["acc"][other_theme]
            ira = aux_acc / (len(in_domain) - 1)

            # evaluate outside the domain
            data = torch.load(os.path.join(path, file_outdomain), map_location=torch.device('cpu'))
            aux_acc = 0.0
            for other_theme in out_domain:
                aux_acc += data["acc"][other_theme]
            cra = aux_acc / len(out_domain)

            # compute fid over all
            fid_file = f"{theme}_fid.pth"
            fid = torch.load(os.path.join(path, fid_file), weights_only=False, map_location=torch.device('cpu'))

            # get memory allocation and time for unlearning
            distill_folder = os.listdir(os.path.join(path_models, f"forget_class_{theme}"))
            if len(distill_folder) > 0:
                distill_f = distill_folder[0]
            else:
                distill_f = distill_folder[0]

            model_data = load_model_card_yaml(os.path.join(path_models, f"forget_class_{theme}", distill_f, "README.md"))

            runtime = retrieve_model_metric(model_data, "Runtime training seconds")
            memory = retrieve_model_metric(model_data, "Peak memory usage in training")

            result_table_c["Theme/Class"].append(theme)
            result_table_c["UA"].append(1.0 - to_float(ua))
            result_table_c["IRA"].append(to_float(ira))
            result_table_c["CRA"].append(to_float(cra))
            result_table_c["FID"].append(to_float(fid))
            result_table_c["Runtime"].append(float(runtime))
            result_table_c["Run_memory"].append(float(memory))
    
    print("\n\nClasses evaluation:\n")
    pretty_table(result_table_c)

    print("\n\nAverages Classes:\n")
    print(f"UA: {sum(result_table_c['UA'])/len(result_table_c['UA'])}")
    print(f"IRA: {sum(result_table_c['IRA'])/len(result_table_c['IRA'])}")
    print(f"CRA: {sum(result_table_c['CRA'])/len(result_table_c['CRA'])}")
    print(f"FID: {sum(result_table_c['FID'])/len(result_table_c['FID'])}")
    print(f"Runtime: {sum(result_table_c['Runtime'])/len(result_table_c['Runtime'])}")
    print(f"Run Memory: {sum(result_table_c['Run_memory'])/len(result_table_c['Run_memory'])}")

    print("\n\nMerged\n")
    merged = {k: result_table_s[k] + result_table_c[k] for k in result_table_s}
    pretty_table(merged)

    print("\n\nAverages:\n")
    print(f"UA: {sum(merged['UA'])/len(merged['UA'])}")
    print(f"IRA: {sum(merged['IRA'])/len(merged['IRA'])}")
    print(f"CRA: {sum(merged['CRA'])/len(merged['CRA'])}")
    print(f"FID: {sum(merged['FID'])/len(merged['FID'])}")
    print(f"Runtime: {sum(merged['Runtime'])/len(merged['Runtime'])}")
    print(f"Run Memory: {sum(merged['Run_memory'])/len(merged['Run_memory'])}")

    # add averages and save tables
    result_table_s["Theme/Class"].append("Averages")
    result_table_s["UA"].append(sum(result_table_s["UA"])/len(result_table_s["UA"]))
    result_table_s["IRA"].append(sum(result_table_s["IRA"])/len(result_table_s["IRA"]))
    result_table_s["CRA"].append(sum(result_table_s["CRA"])/len(result_table_s["CRA"]))
    result_table_s["FID"].append(sum(result_table_s["FID"])/len(result_table_s["FID"]))
    result_table_s["Runtime"].append(sum(result_table_s["Runtime"])/len(result_table_s["Runtime"]))
    result_table_s["Run_memory"].append(sum(result_table_s["Run_memory"])/len(result_table_s["Run_memory"]))

    style_res = pd.DataFrame(result_table_s)
    style_res.to_csv(os.path.join(path, "style_results.csv"), index=False)

    result_table_c["Theme/Class"].append("Averages")
    result_table_c["UA"].append(sum(result_table_c["UA"])/len(result_table_c["UA"]))
    result_table_c["IRA"].append(sum(result_table_c["IRA"])/len(result_table_c["IRA"]))
    result_table_c["CRA"].append(sum(result_table_c["CRA"])/len(result_table_c["CRA"]))
    result_table_c["FID"].append(sum(result_table_c["FID"])/len(result_table_c["FID"]))
    result_table_c["Runtime"].append(sum(result_table_c["Runtime"])/len(result_table_c["Runtime"]))
    result_table_c["Run_memory"].append(sum(result_table_c["Run_memory"])/len(result_table_c["Run_memory"]))

    class_res = pd.DataFrame(result_table_c)
    class_res.to_csv(os.path.join(path, "class_results.csv"), index=False)
    
    merged["Theme/Class"].append("Averages")
    merged["UA"].append(sum(merged["UA"])/len(merged["UA"]))
    merged["IRA"].append(sum(merged["IRA"])/len(merged["IRA"]))
    merged["CRA"].append(sum(merged["CRA"])/len(merged["CRA"]))
    merged["FID"].append(sum(merged["FID"])/len(merged["FID"]))
    merged["Runtime"].append(sum(merged["Runtime"])/len(merged["Runtime"]))
    merged["Run_memory"].append(sum(merged["Run_memory"])/len(merged["Run_memory"]))

    merged_res = pd.DataFrame(merged)
    merged_res.to_csv(os.path.join(path, "merged_results.csv"), index=False)