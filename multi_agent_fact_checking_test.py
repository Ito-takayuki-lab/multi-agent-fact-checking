__author__ = "Dong Yihan"

import requests
import pandas as pd

if __name__ == "__main__":

    # 実験用Liarのデータセットから投稿を読み込む
    # dataset_index = 0
    # df = pd.read_csv("./test_datasets/Liar/liar_modified_2.csv")
    # df = df.dropna()
    # # df = df[df["results"] == "{'detail': 'error in multi_agent_fact_checking()'}"]
    # posts_list = df["claim"].values.tolist()

    # 実験用X-factのデータセットから投稿を読み込む
    # dataset_index = 0
    # df = pd.read_csv("./test_datasets/X-fact_modified/X-fact_modified_all_fair_2.csv")
    # df = df.dropna()
    # posts_list = df["claim"].values.tolist()

    # # 実験用Politifactのデータセットから投稿を読み込む
    # dataset_index = 0
    # df = pd.read_csv("./test_datasets/Politifact_modified/politifact-modified.csv")
    # df = df.dropna()
    # posts_list = df["statement"].values.tolist()

    # # 実験用FEVERのデータセットから投稿を読み込む
    # dataset_index = 0
    # df = pd.read_csv("./test_datasets/FEVER/dev_adjusted.csv")
    # df = df.dropna()
    # posts_list = df["claim"].values.tolist()

    # 実験用SciFactのデータセットから投稿を読み込む
    dataset_index = 0
    df = pd.read_csv("./test_datasets/SciFact/sci_fact.csv")
    df = df.dropna()
    posts_list = df["claim"].values.tolist()

    # with open("./test_datasets/X-fact_modified/temp_dataset_LLM_confidence.txt", "r") as temp_dataset_reader:
    #     temp_dataset = temp_dataset_reader.readlines()
    # posts_list = [data.replace("\n", "") for data in temp_dataset]

    # print(len(posts_list))

    # 投稿を一つずつシステムに投げて、判断結果をリストとして保存する
    temp_results_save_dir = "./experiment_results/SciFact/square_root_test.txt"
    start_index = 0
    end_index = len(posts_list)

    demo_string = ("Trump won the 2016 United States presidential election. "
                   "Trump also won the 2020 United States presidential election.")  # デモに使うテキスト

    for i in range(0, end_index):
        user_post = posts_list[i]

        print(user_post)
        payload = {
            "raw": user_post
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        res = requests.post("http://0.0.0.0:8000/multi_agent/fact_checking_binary", json=payload)
        res = res.json()
        print(res)
        print("The index of the posts list for now:", i + 1)

    #     with open(temp_results_save_dir, mode="a", encoding="utf-8") as temp_results:
    #         temp_results.write(str(res) + "\n")
    #
    # with open(temp_results_save_dir, mode='r') as temp_results_reader:
    #     result_list = temp_results_reader.readlines()
    #
    # result_list = [result.replace("\n", "") for result in result_list]
    # df["results"] = result_list
    # df.to_csv(path_or_buf="./experiment_results/SciFact/ablation_experiments.csv".format(
    #     str(dataset_index)), index=False)
