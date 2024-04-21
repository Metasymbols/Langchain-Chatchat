import sys
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import json, httpx
from typing import List, Dict
from configs import logger, log_verbose


class GeminiWorker(ApiModelWorker):
    """
    GeminiWorker 类，继承自 ApiModelWorker，用于与 Gemini 模型进行交互。

    参数:
    - controller_addr: str, 控制器地址，默认为 None。
    - worker_addr: str, 工作器地址，默认为 None。
    - model_names: List[str], 模型名称列表，默认为 ["gemini-api"]。
    - **kwargs, 其他关键字参数。
    """

    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["gemini-api"],
            **kwargs,
    ):
        # 初始化参数，并调用父类构造函数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        super().__init__(**kwargs)

    def create_gemini_messages(self, messages) -> json:
        """
        根据输入消息创建 Gemini 模型所需的格式。

        参数:
        - messages: 消息列表。

        返回:
        - 转换后的消息的 json 对象。
        """
        has_history = any(msg['role'] == 'assistant' for msg in messages)
        gemini_msg = []

        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                continue
            if has_history:
                if role == 'assistant':
                    role = "model"
                transformed_msg = {"role": role, "parts": [{"text": content}]}
            else:
                if role == 'user':
                    transformed_msg = {"parts": [{"text": content}]}

            gemini_msg.append(transformed_msg)

        msg = dict(contents=gemini_msg)
        return msg

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        与 Gemini 模型进行对话。

        参数:
        - params: ApiChatParams 对象，包含对话所需的参数。

        返回:
        - 与模型交互的结果。
        """
        params.load_config(self.model_names[0])
        data = self.create_gemini_messages(messages=params.messages)
        # 设置生成配置
        generationConfig = dict(
            temperature=params.temperature,
            topK=1,
            topP=1,
            maxOutputTokens=4096,
            stopSequences=[]
        )

        data['generationConfig'] = generationConfig
        # 构造请求URL和头部
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent" + '?key=' + params.api_key
        headers = {
            'Content-Type': 'application/json',
        }
        # 如果设置允许日志详细输出，则记录日志
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')
            logger.info(f'{self.__class__.__name__}:data: {data}')

        text = ""
        json_string = ""
        timeout = httpx.Timeout(60.0)
        client = get_httpx_client(timeout=timeout)
        with client.stream("POST", url, headers=headers, json=data) as response:
            for line in response.iter_lines():
                line = line.strip()
                if not line or "[DONE]" in line:
                    continue

                json_string += line

            try:
                resp = json.loads(json_string)
                if 'candidates' in resp:
                    for candidate in resp['candidates']:
                        content = candidate.get('content', {})
                        parts = content.get('parts', [])
                        # 组装回复文本
                        for part in parts:
                            if 'text' in part:
                                text += part['text']
                                yield {
                                    "error_code": 0,
                                    "text": text
                                }
                        print(text)
            except json.JSONDecodeError as e:
                print("Failed to decode JSON:", e)
                print("Invalid JSON string:", json_string)

    def get_embeddings(self, params):
        """
        获取嵌入表示，当前方法为空，仅为示例。

        参数:
        - params: 参数对象，示例中未使用。
        """
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。

        参数:
        - conv_template: str, 对话模板字符串，默认为 None。
        - model_path: str, 模型路径，默认为 None。

        返回:
        - Conversation 对象。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="You are a helpful, respectful and honest assistant.",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.base_model_worker import app

    worker = GeminiWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21012",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21012)