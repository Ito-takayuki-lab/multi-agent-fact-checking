__author__ = "Dong Yihan"

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objs as go
import plotly.express as px
from matplotlib import pyplot as plt


results_df = pd.read_csv("./experiment_results/SciFact/weighted_score_experiment.csv")
print(results_df)
# ground truthは元のデータセットにのラベルです
ground_truth = results_df["label"].values.tolist()
# verification resultsは実験の結果です
verification_results = results_df["results"].values.tolist()
verification_results = [str(verification_result) for verification_result in verification_results]
verification_results = [verification_result.replace("True", "TRUE") for verification_result in verification_results]
verification_results = [verification_result.replace("False", "FALSE") for verification_result in verification_results]
# verification_results = [verification_result.replace("partly-TRUE", "partly true") for verification_result in verification_results]

ground_truth = [result.replace("REFUTES", "FALSE") for result in ground_truth]
ground_truth = [result.replace("SUPPORTS", "TRUE") for result in ground_truth]
print("unique verification results", list(dict.fromkeys(verification_results)))
print("unique ground truth", list(dict.fromkeys(ground_truth)))

# ground truth binaryは二項分類で処理した元のデータセットにのラベルです。
# ground_truth_binary = [s.replace("mostly-true", "TRUE") for s in ground_truth]
# ground_truth_binary = [s.replace("half-true", "FALSE") for s in ground_truth_binary]
# ground_truth_binary = [s.replace("barely-true", "FALSE") for s in ground_truth_binary]

# 三項分類の場合
ground_truth_binary = [s.replace("mostly-true", "partly true") for s in ground_truth]
ground_truth_binary = [s.replace("half-true", "partly true") for s in ground_truth_binary]
ground_truth_binary = [s.replace("barely-true", "partly true") for s in ground_truth_binary]

# verification results binaryは二項分類で処理した実験の結果です。
# verification_results_binary = [s.replace("mostly-true", "TRUE") for s in verification_results]
# verification_results_binary = [s.replace("half-true", "FALSE") for s in verification_results_binary]
# verification_results_binary = [s.replace("barely-true", "FALSE") for s in verification_results_binary]

# 三項分類の場合
verification_results_binary = [s.replace("mostly-true", "PARTLY") for s in verification_results]
verification_results_binary = [s.replace("half-true", "PARTLY") for s in verification_results_binary]
verification_results_binary = [s.replace("barely-true", "PARTLY") for s in verification_results_binary]

print("unique binary verification results", list(dict.fromkeys(verification_results)))
print("unique binary ground truth", list(dict.fromkeys(ground_truth)))

verification_report = classification_report(ground_truth, verification_results, output_dict=True)
plt.figure(figsize=(5, 2))
result_plot = sns.heatmap(pd.DataFrame(verification_report).iloc[:-1, :].T, annot=True, cmap="Blues")
result_graph = result_plot.get_figure()
result_graph.savefig("./experiment_results/SciFact/Figure/weighted_score_experiment.png", bbox_inches="tight")
# df_for_plot = pd.DataFrame(verification_report).iloc[:-1, :].T.iloc[::-1]
# cell_size = 30
# heatmap = go.Figure()
# heatmap.add_trace(go.Heatmap(z=df_for_plot.values, x=df_for_plot.columns, y=df_for_plot.index, colorscale='Blues'))
# heatmap.update_layout(
#     autosize=False,
#     width=cell_size * 16,
#     height=cell_size * 12,
# )
# for i, row in enumerate(df_for_plot.values):
#     for j, value in enumerate(row):
#         heatmap.add_annotation(
#             text=f"{value:.2f}",  # 小数点以下2桁まで表示
#             x=df_for_plot.columns[j],
#             y=df_for_plot.index[i],
#             xref='x',
#             yref='y',
#             showarrow=False,
#             font=dict(color='white' if value > 0.5 else 'black')  # 任意の条件で文字色を変更できます
#         )
# heatmap.write_image("./experiment_results/FEVER/Figure/heatmap_binary_gpt4.png")
