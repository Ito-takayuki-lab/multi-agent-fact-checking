__author__ = "Dong Yihan"

import json
from abc import abstractmethod
from sentence_transformers import SentenceTransformer
from bert_score import score

from dependencies import language_model


class BaseAgent:
    """
    異なるファクトチェックエージェントに共通のパラメーターやメソッドを定義するため、BaseAgent（基盤エージェント）の定義です。
    """

    def __init__(self, agent_name, agent_weight):
        self.status = {
            "status": 500,
            "message": "error in initializing BaseAgent"
        }
        """
        主張を検証するためのプロンプト
        初期化メソッドには、エージェント名とウェイトが異なるエージェントによって変わるですが、主張を検証するためのプロンプトが変わらない。
        """
        with open("./prompts/agreement_verification_with_relevance.json", "r") as claim_verification_file:
            verification_data = json.load(claim_verification_file)
        self.claim_verification_instruct = {
            "system": verification_data["system"],
            "user": verification_data["user"]
        }
        with open("./prompts/evaluate_relevance.json", "r") as relevance_file:
            relevance_data = json.load(relevance_file)
        self.relevance_instruct = {
            "system": relevance_data["system"],
            "user": relevance_data["user"]
        }
        # エージェント名
        self.agent_name = agent_name
        # エージェントのウェイト、デフォルトは1
        self.weight = agent_weight
        # エージェントの言語モデル
        self.openai = language_model.OPENAI()
        # エージェントのembeddingモデル
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.status.update(status=200, message="successfully initialize BaseAgent")
        return

    # 主張の正しさを判断するメソッド。ここのメソッドはテンプレートとして使う
    @abstractmethod
    def claims_verification(self):
        return

    # # falseと判断された主張と理由を抽出するメソッド
    # # verification_resultsには全ての主張の正しさを含み、そしてverification_resultsとclaimsの長さが同じはずだ
    # def extract_false_claims_and_reasons(self, verification_results, claims):
    #     self.status.update(status=500, message="error in extract_false_claims_and_reasons"
    #                                            " of the agent {}".format(self.agent_name))
    #
    #     false_claims_and_reasons = ""
    #     if len(verification_results) != len(claims):
    #         self.status.update(status=500, message="verification_results and claims have no same length"
    #                                                " of the agent {}".format(self.agent_name))
    #         return false_claims_and_reasons
    #     for i in range(0, len(verification_results)):
    #         if not verification_results[i]["factuality"]:
    #             false_claims_and_reasons = false_claims_and_reasons + "> " + claims[i]["claim"] + \
    #                                        "\n\nis probably not correct.\n" + verification_results[i]["reasoning"] + \
    #                                        "\n\n"
    #         self.status.update(status=200, message="successfully extract the false claims and reasons")
    #
    #     return false_claims_and_reasons

    def calculate_bert_score(self, claim: str, evidence: str):
        """
        主張と検索した情報（つまりevidence）のBERTScoreを計算して、情報と主張の関係度を示す
        :param claim: str　主張
        :param evidence: str 検索された情報
        :return: bert_score_f1: float BERTScoreのF1値
        """
        self.status.update(status=500, message="error in calculate_cosine_similarity of base agent")

        precision, recall, f1 = score(cands=[claim], refs=[evidence], lang="en", verbose=True,
                                      model_type="microsoft/deberta-large-mnli")
        bert_score_f1 = f1.tolist()[0]

        self.status.update(status=200, message="successfully calculate cosine similarity of the claim and the evidence")
        return bert_score_f1

    def evaluate_relevance(self, claim_evidence_list):
        """
        主張と検索された情報の関連度を値踏みするメソッド
        :param claim_evidence_list: list[dict{"claim": claim, "evidence": "evidence"}] 主張と情報が共に保存されたリスト
        :return: relevance: str 主張と情報との関連度
        """
        self.status.update(status=500, message="error in evaluate_relevance of base agent")

        relevance_responses = []
        for i in range(0, len(claim_evidence_list)):
            claim = claim_evidence_list[i]["claim"]
            evidence = claim_evidence_list[i]["evidence"]
            relevance_evaluation_prompts = [
                {
                    "role": "system",
                    "content": self.relevance_instruct["system"]
                },
                {
                    "role": "user",
                    "content": self.relevance_instruct["user"].format(
                        claim=claim, evidence=evidence
                    )
                }
            ]
            relevance_response = self.openai.get_gpt_response(prompts=relevance_evaluation_prompts)

            print("relevance response: ", relevance_response)
            relevance_responses.append(relevance_response)

        self.status.update(status=200, message="successfully evaluate the relevance of claim and evidence")
        return relevance_responses

    def combine_confidence_with_claims(self, verification_results, claims):
        """
        主張、確信度及び判断結果を組み合わせるメソッドです。
        :param verification_results: 言語モデルによりの判断結果です。list[dict{}]
        :param claims: 抽出された主張リスト。list[dict{"claim": str}]
        :return: claims_and_confidence: list[dict{"claim", "confidence", "factuality"}]　各エージェントの判断結果です。
        """
        self.status.update(status=500, message="error in calculate_confidence_of_claims"
                                               " of the agent {}".format(self.agent_name))

        claims_and_confidence = []

        # もしエージェントの判断結果の長さと元の主張リストの長さは違いでしたら、エラーコードを戻す。
        if len(verification_results) != len(claims):
            self.status.update(status=500, message="verification_results and claims have no same length"
                                                   " of the agent {}".format(self.agent_name))
        else:
            for i in range(0, len(verification_results)):

                print(verification_results[i]["factuality"])
                # 3.5-turbo以外のモデルを使えば、ちゃんとタイプを変換できない場合があります。
                if type(verification_results[i]["factuality"]) is str:
                    if verification_results[i]["factuality"] == "True" or verification_results[i]["factuality"] == "False":
                        verification_results_boolean = eval(verification_results[i]["factuality"])
                    elif verification_results[i]["factuality"] == "TRUE":
                        verification_results_boolean = True
                    elif verification_results[i]["factuality"] == "FALSE":
                        verification_results_boolean = False
                else:
                    verification_results_boolean = verification_results[i]["factuality"]

                # 本番処理
                if verification_results_boolean is True:
                    confidence = verification_results[i]["score"]
                else:
                    confidence = verification_results[i]["score"] * -1

                # ここに変数confidenceの名前を考え直す必要があるかもしれません
                claim_and_confidence = {
                    "claim": claims[i]["claim"],
                    "confidence": confidence,
                    "factuality": verification_results_boolean
                }
                claims_and_confidence.append(claim_and_confidence)
            self.status.update(status=200, message="successfully calculate the confidence of claims "
                                                   "according to the agent {}".format(self.agent_name))

        print("claims and confidence: ", claims_and_confidence)
        return claims_and_confidence
