__author__ = "Dong Yihan"

import json
from .base_agent import BaseAgent


class LLMJudgementAgent(BaseAgent):
    """
    提案手法を修正するため、言語モデルに基づく判断エージェントを作成する
    判断エージェントがMAFC Scoreを参照として扱い、元の文章の信憑性ラベルを判断する
    比較実験はMAFC Scoreではなく直接に各エージェントの判断結果と信頼性（つまりconfidence）を参照した上で、信憑性ラベルを判断する。
    """

    def claims_verification(self):
        pass

    def __init__(self):
        """
        判断エージェントの初期化です。
        credibility_verification_single_agent_with_MAFCというファイルに保存されたプロンプトを使う。
        """
        super().__init__(agent_name="judgement_agent", agent_weight=1.0)
        with (open("./prompts/credibility_verification_single_agent_with_MAFC.json", "r") as
              credibility_verification_file):
            verification_data = json.load(credibility_verification_file)
        self.credibility_verification_instruct = {
            "system": verification_data["system"],
            "user": verification_data["user"]
        }

        self.status.update(status=200, message="successfully initialize judgement agent")
        return

    def text_credibility_verification(self, text: str):
        """
        元のテキストの信ぴょう性ラベルを判断するメソッド。

        :param text: 元のテキスト
        :return: credibility_label: str. True/partly-true/False
        """

        credibility_label = ""
        verification_prompts = [
            {
                "role": "system",
                "content": self.credibility_verification_instruct["system"]
            },
            {
                "role": "user",
                "content": self.credibility_verification_instruct["user"].format(text=text)
            }
        ]

        judgement_result = self.openai.get_gpt_response(prompts=verification_prompts)
        judgement_result = judgement_result.replace("True", "TRUE")
        judgement_result = judgement_result.replace("False", "FALSE"),
        judgement_result = judgement_result.replace("partly-TRUE", "partly-true")

        print("judgement result:", judgement_result)
        judgement_result_dict = eval(judgement_result)
        credibility_label = judgement_result_dict["factuality"]

        self.status.update(state=200, message="successfully judge credibility label")
        return credibility_label

