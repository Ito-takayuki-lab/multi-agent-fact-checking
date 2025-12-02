__author__ = "Dong Yihan"

import requests
import pandas as pd

"""
シングルなエージェントの判断結果により、簡単な投票でマルチエージェント事実検証を行う場合にこちらのファイルを起動
"""

if __name__ == "__main__":
    # 各エージェントの判断結果を読み込む
    df = pd.read_csv("./test_datasets/FEVER/dev_adjusted.csv")
    df = df.dropna()
    df_google = pd.read_csv("./experiment_results/FEVER/google_agent.csv")
    df_wiki = pd.read_csv("./experiment_results/FEVER/wiki_agent.csv")
    df_news = pd.read_csv("./experiment_results/FEVER/news_agent.csv")

    google_results = df_google["results"].values.tolist()
    wiki_results = df_wiki["results"].values.tolist()
    news_results = df_news["results"].values.tolist()

    voting_results = []

    # 簡単な投票の場合
    for i in range(0, len(google_results)):
        agents_results = [google_results[i], wiki_results[i], news_results[i]]
        right_agents_number = len([r for r in agents_results if r is True])
        wrong_agents_number = 3 - right_agents_number
        if right_agents_number > wrong_agents_number:
            voting_result = True
        else:
            voting_result = False
        voting_results.append(voting_result)

    df["results"] = voting_results
    df.to_csv("./experiment_results/FEVER/simple_voting_binary.csv")
