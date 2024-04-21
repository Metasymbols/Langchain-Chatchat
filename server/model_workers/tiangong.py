import json
import time
import hashlib

from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import json
from typing import List, Literal, Dict
import requests


class TianGongWorker(ApiModelWorker):
    """
    天工API模型工作者类，继承自ApiModelWorker，用于与天工智能聊天API进行交互。

    参数:
    - controller_addr: str, 控制器地址，默认为None。
    - worker_addr: str, 工作者地址，默认为None。
    - model_names: List[str], 模型名称列表，默认为["tiangong-api"]。
    - version: Literal["SkyChat-MegaVerse"], 版本号，默认为"SkyChat-MegaVerse"。
    """

    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["tiangong-api"],
            version: Literal["SkyChat-MegaVerse"] = "SkyChat-MegaVerse",
            **kwargs,
    ):
        # 更新初始化参数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 32768)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        与天工API进行聊天交互。

        参数:
        - params: ApiChatParams, 聊天参数对象，包含必要的配置信息。

        返回:
        - Dict, 包含聊天响应的结果。
        """
        params.load_config(self.model_names[0])

        # 构建请求参数
        url = 'https://sky-api.singularity-ai.com/saas/api/v4/generate'
        data = {
            "messages": params.messages,
            "model": "SkyChat-MegaVerse"
        }
        timestamp = str(int(time.time()))
        sign_content = params.api_key + params.secret_key + timestamp
        sign_result = hashlib.md5(sign_content.encode('utf-8')).hexdigest()
        headers = {
            "app_key": params.api_key,
            "timestamp": timestamp,
            "sign": sign_result,
            "Content-Type": "application/json",
            "stream": "true"  # 启用或禁用流式返回内容处理
        }

        # 发起请求并处理响应
        response = requests.post(url, headers=headers, json=data, stream=True)

        text = ""
        # 处理解码后的响应流
        for line in response.iter_lines(chunk_size=None, decode_unicode=True):
            if line:
                resp = json.loads(line)
                if resp["code"] == 200:
                    # 正常响应，累加回复文本
                    text += resp['resp_data']['reply']
                    yield {
                        "error_code": 0,
                        "text": text
                    }
                else:
                    # 错误处理
                    data = {
                        "error_code": resp["code"],
                        "text": resp["code_msg"]
                    }
                    self.logger.error(f"请求天工 API 时出错：{data}")
                    yield data

    def get_embeddings(self, params):
        """
        获取嵌入表示。

        参数:
        - params:，当前方法为空实现，用于未来扩展 输入参数，具体类型和结构依据实际需求确定。
        """
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。

        参数:
        - conv_template: str, 对话模板字符串，默认为None。
        - model_path: str, 模型路径，默认为None。

        返回:
        - Conversation, 初始化后的对话对象。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "system"],
            sep="\n### ",
            stop_str="###",
        )