import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("table_UC.csv")

methods = df["Method"].unique()

# Create color map (one color per method)
cmap = plt.get_cmap("tab20")
colors = {m: cmap(i % 20) for i, m in enumerate(methods)}

# Font sizes
LABEL_SIZE = 14
TITLE_SIZE = 16
TICK_SIZE = 12
LEGEND_SIZE = 12

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# ---------- UA vs IRA ----------
ax = axes[0]
for _, row in df.iterrows():

    if row["Method"] == "Ours":
        size = 100
    else:
        size = 70

    ax.scatter(
        row["Style_IRA"], row["Style_UA"],
        marker="o",
        color=colors[row["Method"]],
        s=size, edgecolors="black", linewidths=0.5
    )
    ax.scatter(
        row["Object_IRA"], row["Object_UA"],
        marker="^",
        color=colors[row["Method"]],
        s=size, edgecolors="black", linewidths=0.5
    )

ax.set_xlabel("IRA (%)", fontsize=LABEL_SIZE)
ax.set_ylabel("UA (%)", fontsize=LABEL_SIZE)
ax.set_title("UA vs IRA", fontsize=TITLE_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.grid(True)

# ---------- UA vs CRA ----------
ax = axes[1]
for _, row in df.iterrows():

    if row["Method"] == "Ours":
        size = 100
    else:
        size = 70
        
    ax.scatter(
        row["Style_CRA"], row["Style_UA"],
        marker="o",
        color=colors[row["Method"]],
        s=size, edgecolors="black", linewidths=0.5
    )
    ax.scatter(
        row["Object_CRA"], row["Object_UA"],
        marker="^",
        color=colors[row["Method"]],
        s=size, edgecolors="black", linewidths=0.5
    )

ax.set_xlabel("CRA (%)", fontsize=LABEL_SIZE)
ax.set_title("UA vs CRA", fontsize=TITLE_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.grid(True)

# ---------- Legends ----------
method_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=color)
    for color in colors.values()
]
method_labels = list(colors.keys())

fig.legend(
    method_handles,
    method_labels,
    loc="center right",
    title="Method",
    title_fontsize=LEGEND_SIZE,
    fontsize=LEGEND_SIZE
)

domain_handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color='black', label='Style'),
    plt.Line2D([0], [0], marker='^', linestyle='', color='black', label='Object')
]
fig.legend(
    domain_handles,
    ["Style", "Object"],
    loc="lower right",
    fontsize=LEGEND_SIZE
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig("UA_IRA_CRA_scatter_plots.png", dpi=300)

# ---------- TIME vs MEMORY ----------
plt.figure(figsize=(7, 5))

for _, row in df.iterrows():
    if pd.isna(row["Time_s"]) or pd.isna(row["Memory_GB"]):
        continue

    if row["Method"] == "Ours":
        size = 100
    else:
        size = 70

    plt.scatter(
        row["Time_s"],
        row["Memory_GB"],
        color=colors[row["Method"]],
        s=size,
        edgecolors="black",
        linewidths=0.5
    )

plt.xlabel("Time (s)", fontsize=LABEL_SIZE)
plt.ylabel("Peak GPU Memory (GB)", fontsize=LABEL_SIZE)
plt.title("Time vs Memory", fontsize=TITLE_SIZE)
plt.tick_params(axis="both", labelsize=TICK_SIZE)
plt.grid(True)

# Legend outside the plot
handles = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=color)
    for color in colors.values()
]
labels = list(colors.keys())

plt.legend(
    handles,
    labels,
    title="Method",
    title_fontsize=LEGEND_SIZE,
    fontsize=LEGEND_SIZE,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)  # x=1.02 → just outside the plot
)

plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space on the right for legend
plt.savefig("Time_Memory_scatter_plot.png", dpi=300)