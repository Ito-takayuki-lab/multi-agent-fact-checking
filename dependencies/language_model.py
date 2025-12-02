__author__ = "Dong Yihan"

import openai
import tiktoken


class OPENAI:

    """
    openaiの言語モデルを用いる場合には、パラメーターや設置などを初期化するメソッド。
    """
    def __init__(self):
        self.status = {
            "status": 500,
            "message": "error in initialization of OPENAI"
        }

        # openaiのapiを使う時にのkeyです
        self.openai_key = ""

        # openaiの言語モデルについての設定
        self.gpt_config = {
            'model_name': 'gpt-4o',
            'temperature': 0.0,
            'top_p': 1,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'n': 1
        }

        # プロンプトの長さを測るため、tiktokenというライブラリーを使う
        self.encoding = tiktoken.encoding_for_model(self.gpt_config["model_name"])

        self.status.update(status=200, message="successfully initialize class openai")
        return

    def get_gpt_response(self, prompts: list[dict[str, str]]):
        """
        gptから返事を得るメソッド。

        gptに投げるプロンプトはパラメーターであり、返事の文章は戻すです。
        :param prompts: list[dict["role": "system/user/assistant", "content"]]
        :return:
        """
        self.status.update(status=500, message="error in get_gpt_response()")

        openai.api_key = self.openai_key
        response = openai.chat.completions.create(
            model=self.gpt_config["model_name"],
            messages=prompts,
            temperature=self.gpt_config["temperature"],
            top_p=self.gpt_config["top_p"],
            frequency_penalty=self.gpt_config["frequency_penalty"],
            presence_penalty=self.gpt_config["presence_penalty"],
            n=self.gpt_config["n"]
        )
        response_content = response.choices[0].message.content

        return response_content
