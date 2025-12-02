__author__ = "Dong Yihan"

import pandas as pd
import matplotlib.pyplot as plt

# confidence_data_dir = "./experiment_results/SciFact/weighted_score_save_dir.txt"
# data = pd.read_csv(confidence_data_dir, sep=" ", header=None, index_col=False)
#
confidence_data_csv_dir = "./experiment_results/SciFact/weighted_score_save_dir.csv"
# data.to_csv(confidence_data_csv_dir, index=True, header=["Agent1", "Agent2", "Agent3",
#                                                          "Right Confidence", "Wrong Confidence", "Real Confidence",
#                                                          "Normalized Confidence"])

data_csv = pd.read_csv(confidence_data_csv_dir)
data_csv["Average Confidence"] = (data_csv["Agent1"] + data_csv["Agent2"] + data_csv["Agent3"]) / 3
print(data_csv)

# fig, axes = plt.subplots(3, 1, sharex="all")
# axes[0].plot(data_csv["Unnamed: 0"], data_csv["Agent1"], color="red", linestyle="dotted", label="Agent 1")
# axes[0].plot(data_csv["Unnamed: 0"], data_csv["Agent2"], color="green", linestyle="dotted", label="Agent 2")
# axes[0].plot(data_csv["Unnamed: 0"], data_csv["Agent3"], color="blue", linestyle="dotted", label="Agent 3")
# axes[0].legend(loc="best")
# axes[1].plot(data_csv["Unnamed: 0"], data_csv["Right Confidence"], color="tomato", linestyle="dashdot",
#              label="Correctness Score")
# axes[1].plot(data_csv["Unnamed: 0"], data_csv["Wrong Confidence"], color="lime", linestyle="dashdot",
#              label="Incorrectness Score")
# axes[1].plot(data_csv["Unnamed: 0"], data_csv["Real Confidence"], color="c", linestyle="dashdot",
#              label="Veracity Score")
# axes[1].legend(loc="best")
# axes[2].plot(data_csv["Unnamed: 0"], data_csv["Normalized Confidence"], color="salmon", linestyle="solid",
#              label="Normalized Score")
# axes[2].plot(data_csv["Unnamed: 0"], data_csv["Average Confidence"], color="cyan", linestyle="solid",
#              label="Average Confidence")
# axes[2].legend(loc="best")
# axes[0].xaxis.set_major_locator(plt.MultipleLocator(1))
# axes[1].xaxis.set_major_locator(plt.MultipleLocator(1))
# axes[2].xaxis.set_major_locator(plt.MultipleLocator(1))
# axes[0].set_xlabel("Claims of Experiments")
# axes[1].set_xlabel("Claims of Experiments")
# axes[2].set_xlabel("Claims of Experiments")
# axes[0].set_ylabel("Confidence")
# axes[1].set_ylabel("Confidence")
# axes[2].set_ylabel("Confidence")
# axes[0].set_title("Confidence of Each Agent in Experiments")
# axes[1].set_title("Confidence of Proposed Method")
# axes[2].set_title("Confidence of Proposed Method")
# axes[0].axis(xmin=-1, xmax=188)
# axes[1].axis(xmin=-1, xmax=188)
# axes[2].axis(xmin=-1, xmax=188)
# fig.set_figwidth(28)
# fig.set_figheight(8)
# fig.autofmt_xdate()
# fig.show()
# fig.savefig("./experiment_results/SciFact/weighted_score.png", format="png", dpi=1000)

# fig, ax = plt.subplots(1, 1)
# ax.scatter(data_csv["Unnamed: 0"], data_csv["Agent1"], color="red", s=10, label="Agent 1")
# ax.scatter(data_csv["Unnamed: 0"], data_csv["Agent2"], color="green", s=10, label="Agent 2")
# ax.scatter(data_csv["Unnamed: 0"], data_csv["Agent3"], color="blue", s=10, label="Agent 3")
# ax.scatter(data_csv["Unnamed: 0"], data_csv["Normalized Confidence"], color="gold", s=10,
#               label="Normalized Confidence")
# ax.legend(loc="best")
# ax.axis(xmin=0, xmax=188, ymin=-1.1, ymax=1.1)
# fig.set_figwidth(20)
# fig.set_figheight(8)
# fig.autofmt_xdate()
# fig.show()
# fig.savefig("./experiment_results/SciFact/scatted_confidence.png", format="png", dpi=700)


fig, ax = plt.subplots(1, 1)
ax.boxplot(data_csv[["Agent1", "Agent2", "Agent3"]].values)
ax.set_title("Agents' confidence")
ax.set_xlabel("Agent")
ax.set_ylabel("Confidence")
ax.set_xticklabels(["Agent 1", "Agent 2", "Agent 3"])
fig.set_figwidth(10)
fig.set_figheight(7)
fig.show()
fig.savefig("./experiment_results/SciFact/boxplot_confidence.png", format="png", dpi=700)
