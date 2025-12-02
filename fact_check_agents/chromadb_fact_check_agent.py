__author__ = "Dong Yihan"

import chromadb
from .base_agent import BaseAgent


# chromadbに文章を保存した上で、事実を検証するエージェント
class ChromaDBFactCheckAgent(BaseAgent):

    def __init__(self):
        super().__init__(agent_name="chromadb_fact_check_agent", agent_weight=1.0)

        self.status.update(status=200, message="successfully initialize chromadb agent")
        return

    #

    # 主張の正しさを判断するメソッド
    def claims_verification(self):
        pass
