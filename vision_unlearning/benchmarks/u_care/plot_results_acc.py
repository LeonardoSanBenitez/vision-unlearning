import os
import torch
from prettytable import PrettyTable
import yaml
import re
import pandas as pd

from constants import class_available, theme_available


path = "assets/NatysResults/"
path_models = "assets/models/"

def pretty_table(dct):
    table = PrettyTable()
    for c in dct.keys():
        table.add_column(c, [])
    table.add_row(['\n'.join(map(str, dct[c])) for c in dct.keys()])
    print(table)


def dict_to_latex_table(table: dict[str, list], caption: str = "", label: str = "") -> None:
    """
    Print a dictionary as a LaTeX table.
    
    Args:
        table (dict[str, list]): Keys are column names, values are column data.
        caption (str): Optional LaTeX caption.
        label (str): Optional LaTeX label.
    """
    headers = list(table.keys())
    num_cols = len(headers)
    num_rows = len(next(iter(table.values()), []))

    # Begin table
    print("\\begin{table}[h]")
    print("\\centering")
    if caption:
        print(f"\\caption{{{caption}}}")
    if label:
        print(f"\\label{{{label}}}")
    print(f"\\begin{{tabular}}{{{'|'.join(['c'] * num_cols)}}}")
    print("\\hline")

    # Header row
    print(" & ".join(["\\textbf{{{0}}}".format(h) for h in headers]) + " \\\\")
    print("\\hline")

    # print classes
    print("\multicolumn{" + str(len(headers)) + "}{" + "c}{" + "Object Unlearning" + "} \\\\")
    print("\\hline")

    # Data rows
    for i in range(num_rows):
        line_name = str(table[headers[0]][i]).replace('_', " ")
        if table[headers[0]][i] not in class_available:
            continue
        row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
        print(line_name + " & " + " & ".join(row) + " \\\\")
        print("\\hline")
    
    # print averages
    print("\\rowcolor[HTML]{BBF3F1}")

    # Data rows
    for i in range(num_rows):
        line_name = "\\textbf{" + "Average" + "}"
        if table[headers[0]][i] not in ["Average Classes"]:
            continue
        row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
        print(line_name + " & " + " & ".join(row) + " \\\\")
        print("\\hline")
    
    # print Styles
    print("\multicolumn{" + str(len(headers)) + "}{" + "c}{" + "Style Unlearning" + "} \\\\")
    print("\\hline")

    # Data rows
    for i in range(num_rows):
        line_name = str(table[headers[0]][i]).replace('_', " ")
        if table[headers[0]][i] not in theme_available:
            continue
        row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
        print(line_name + " & " + " & ".join(row) + " \\\\")
        print("\\hline")
    
    print("\\rowcolor[HTML]{BBF3F1}")
    
    # Data rows
    for i in range(num_rows):
        line_name = "\\textbf{" + "Average" + "}"
        if table[headers[0]][i] not in ["Average Styles"]:
            continue
        row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
        print(line_name + " & " + " & ".join(row) + " \\\\")
        print("\\hline")

    # End table
    print("\\end{tabular}")
    print("\\end{table}")


def dict_to_latex_table_split(table: dict[str, list], caption: str = "", label: str = "", split=True, class_name:str="") -> None:
    """
    Print a dictionary as a LaTeX table.
    
    Args:
        table (dict[str, list]): Keys are column names, values are column data.
        caption (str): Optional LaTeX caption.
        label (str): Optional LaTeX label.
    """
    headers = list(table.keys())
    num_cols = len(headers)
    num_rows = len(next(iter(table.values()), []))

    if not split:

        # Begin table
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\small")
        if caption:
            print(f"\\caption{{{caption}}}")
        if label:
            print(f"\\label{{{label}}}")
        print(f"\\begin{{tabular}}{{{'|'.join(['c'] * num_cols)}}}")
        print("\\hline")

        # Header row
        print(f"\\textbf{{{class_name}}}" + " & " + " & ".join(["\\textbf{{{0}}}".format(h) for h in headers[1:]]) + " \\\\")
        print("\\hline")

        # Data rows
        for i in range(num_rows):
            line_name = str(table[headers[0]][i]).replace('_', " ")
            if table[headers[0]][i] not in class_available:
                continue
            row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
            print(line_name + " & " + " & ".join(row) + " \\\\")
            print("\\hline")
        
        # print averages
        print("\\rowcolor[HTML]{BBF3F1}")

        # Data rows
        for i in range(num_rows):
            line_name = "\\textbf{" + "Average" + "}"
            if table[headers[0]][i] not in ["Average"]:
                continue
            row = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
            print(line_name + " & " + " & ".join(row) + " \\\\")
            print("\\hline")

        # End table
        print("\\end{tabular}")
        print("\\end{table}")
    
    else:

        num_rows_half = num_rows // 2

        # Begin table
        print("\\begin{table}[h]")
        print("\\centering")
        if caption:
            print(f"\\caption{{{caption}}}")
        if label:
            print(f"\\label{{{label}}}")
        #print("{\\scriptsize")
        print(f"\\begin{{tabular}}{{{'|'.join(['c'] * num_cols) + '||' + '|'.join(['c'] * num_cols)}}}")
        print("\\hline")

        # Header row
        print(f"\\textbf{{{class_name}}}" + " & " + " & ".join(["\\textbf{{{0}}}".format(h) for h in headers[1:]]) + " & " + f"\\textbf{{{class_name}}}" + " & " + " & ".join(["\\textbf{{{0}}}".format(h) for h in headers[1:]]) + " \\\\")
        print("\\hline")

        # Data rows
        for i in range(num_rows_half):
            line_name_one = str(table[headers[0]][i]).replace('_', " ")
            line_name_two = str(table[headers[0]][num_rows_half + i]).replace('_', " ")
            row_one = [f"{(table[h][i]):.2f}" if h == "FID" else f"{(table[h][i]):.2f}\%" for h in headers[1:]]
            row_two = [f"{(table[h][num_rows_half + i]):.2f}" if h == "FID" else f"{(table[h][num_rows_half + i]):.2f}\%" for h in headers[1:]]
            print(line_name_one + " & " + " & ".join(row_one) + " & " + line_name_two + " & " + " & ".join(row_two) + " \\\\")
            print("\\hline")
        
        if num_rows % 2 == 1:

            if table[headers[0]][num_rows - 1].startswith("Average"):
                # print averages
                print("\\rowcolor[HTML]{BBF3F1}")
                line_name_two = "Average"
            else:
                line_name_two = str(table[headers[0]][num_rows - 1]).replace('_', " ")

            # print last row if odd number of rows
            line_name_one = " "
            row_one = [" " for h in headers[1:]]
            row_two = [f"{(table[h][num_rows - 1]):.2f}" if h == "FID" else f"{(table[h][num_rows - 1]):.2f}\%" for h in headers[1:]]
            print(line_name_one + " & " + " & ".join(row_one) + " & " + line_name_two + " & " + " & ".join(row_two) + " \\\\")
            print("\\hline")

        # End table
        print("\\end{tabular}")
        #print("}")
        print("\\end{table}")



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

    result_table_s: dict[str, list] = {"Class/Theme": [], "UA": [], "IRA": [], "CRA": [], "FID": []}

    files = os.listdir(path)

    all_themes = theme_available + class_available

    for theme in theme_available:
        ua = 0.0
        ira = 0.0
        cra = 0.0
        fid = 0.0

        if f"{theme}_style.pth" in files and f"{theme}_class.pth" in files:
            
            # evaluate in the same domain
            if theme in theme_available:
                file_indomain = f"{theme}_style.pth"
                file_outdomain = f"{theme}_class.pth"
                in_domain = theme_available
                out_domain = class_available

            else:
                file_indomain = f"{theme}_class.pth"
                file_outdomain = f"{theme}_style.pth"
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

            result_table_s["Class/Theme"].append(theme)
            result_table_s["UA"].append((1.0 - to_float(ua))*100)
            result_table_s["IRA"].append((to_float(ira))*100)
            result_table_s["CRA"].append((to_float(cra))*100)

            # compute fid over all
            if f"{theme}_fid.txt" in files:
                fid_file = f"{theme}_fid.txt"
                with open(os.path.join(path, fid_file), 'r') as f:
                    fid = float(f.read())
                result_table_s["FID"].append(to_float(fid))
            else:
                result_table_s["FID"].append(0)

    print("\n\nStyles evaluation:\n")
    pretty_table(result_table_s)

    print("\n\nAverages Styles:\n")
    print(f"UA: {sum(result_table_s['UA'])/len(result_table_s['UA'])}")
    print(f"IRA: {sum(result_table_s['IRA'])/len(result_table_s['IRA'])}")
    print(f"CRA: {sum(result_table_s['CRA'])/len(result_table_s['CRA'])}")
    print(f"FID: {sum(result_table_s['FID'])/len(result_table_s['FID'])}")

    result_table_c: dict[str, list] = {"Class/Theme": [], "UA": [], "IRA": [], "CRA": [], "FID": []}
    for theme in class_available:
        ua = 0.0
        ira = 0.0
        cra = 0.0
        fid = 0.0

        if f"{theme}_style.pth" in files and f"{theme}_class.pth" in files:

            # evaluate in the same domain
            if theme in class_available:
                file_indomain = f"{theme}_class.pth"
                file_outdomain = f"{theme}_style.pth"
                in_domain = class_available
                out_domain = theme_available

            else:
                file_indomain = f"{theme}_style.pth"
                file_outdomain = f"{theme}_class.pth"
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

            result_table_c["Class/Theme"].append(theme)
            result_table_c["UA"].append((1.0 - to_float(ua))*100)
            result_table_c["IRA"].append((to_float(ira))*100)
            result_table_c["CRA"].append((to_float(cra))*100)

            # compute fid over all
            if f"{theme}_fid.txt" in files:
                fid_file = f"{theme}_fid.txt"
                with open(os.path.join(path, fid_file), 'r') as f:
                    fid = float(f.read())
                result_table_c["FID"].append(to_float(fid))
            else:
                result_table_c["FID"].append(0)
    
    print("\n\nClasses evaluation:\n")
    pretty_table(result_table_c)

    print("\n\nAverages Classes:\n")
    print(f"UA: {sum(result_table_c['UA'])/len(result_table_c['UA'])}")
    print(f"IRA: {sum(result_table_c['IRA'])/len(result_table_c['IRA'])}")
    print(f"CRA: {sum(result_table_c['CRA'])/len(result_table_c['CRA'])}")
    print(f"FID: {sum(result_table_c['FID'])/len(result_table_c['FID'])}")

    print("\n\nMerged\n")
    merged = {k: result_table_s[k] + result_table_c[k] for k in result_table_s}
    pretty_table(merged)

    print("\n\nAverages:\n")
    print(f"UA: {sum(merged['UA'])/len(merged['UA'])}")
    print(f"IRA: {sum(merged['IRA'])/len(merged['IRA'])}")
    print(f"CRA: {sum(merged['CRA'])/len(merged['CRA'])}")
    print(f"FID: {sum(merged['FID'])/len(merged['FID'])}")

    # add averages and save tables
    result_table_s["Class/Theme"].append("Average Styles")
    result_table_s["UA"].append(sum(result_table_s["UA"])/len(result_table_s["UA"]))
    result_table_s["IRA"].append(sum(result_table_s["IRA"])/len(result_table_s["IRA"]))
    result_table_s["CRA"].append(sum(result_table_s["CRA"])/len(result_table_s["CRA"]))
    result_table_s["FID"].append(sum(result_table_s["FID"])/len(result_table_s["FID"]))

    style_res = pd.DataFrame(result_table_s)
    style_res.to_csv(os.path.join(path, "style_results.csv"), index=False)

    result_table_c["Class/Theme"].append("Average Classes")
    result_table_c["UA"].append(sum(result_table_c["UA"])/len(result_table_c["UA"]))
    result_table_c["IRA"].append(sum(result_table_c["IRA"])/len(result_table_c["IRA"]))
    result_table_c["CRA"].append(sum(result_table_c["CRA"])/len(result_table_c["CRA"]))
    result_table_c["FID"].append(sum(result_table_c["FID"])/len(result_table_c["FID"]))
    
    class_res = pd.DataFrame(result_table_c)
    class_res.to_csv(os.path.join(path, "class_results.csv"), index=False)
    
    merged["Class/Theme"].append("Average All")
    merged["UA"].append(sum(merged["UA"])/len(merged["UA"]))
    merged["IRA"].append(sum(merged["IRA"])/len(merged["IRA"]))
    merged["CRA"].append(sum(merged["CRA"])/len(merged["CRA"]))
    merged["FID"].append(sum(merged["FID"])/len(merged["FID"]))

    merged["Class/Theme"].append("Average Styles")
    merged["UA"].append(sum(result_table_s["UA"])/len(result_table_s["UA"]))
    merged["IRA"].append(sum(result_table_s["IRA"])/len(result_table_s["IRA"]))
    merged["CRA"].append(sum(result_table_s["CRA"])/len(result_table_s["CRA"]))
    merged["FID"].append(sum(result_table_s["FID"])/len(result_table_s["FID"]))

    merged["Class/Theme"].append("Average Classes")
    merged["UA"].append(sum(result_table_c["UA"])/len(result_table_c["UA"]))
    merged["IRA"].append(sum(result_table_c["IRA"])/len(result_table_c["IRA"]))
    merged["CRA"].append(sum(result_table_c["CRA"])/len(result_table_c["CRA"]))
    merged["FID"].append(sum(result_table_c["FID"])/len(result_table_c["FID"]))

    dict_to_latex_table(merged, caption="Merged Results", label="tab:merged_results")

    merged_res = pd.DataFrame(merged)
    merged_res.to_csv(os.path.join(path, "merged_results.csv"), index=False)

    print("\n\nLaTeX Tables:\n")
    dict_to_latex_table_split(result_table_c, caption="Object Unlearning Results", label="tab:object_unlearning_results", split=True, class_name="Class")
    print('\n\n')
    dict_to_latex_table_split(result_table_s, caption="Style Unlearning Results", label="tab:style_unlearning_results", split=True, class_name="Theme")