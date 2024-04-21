import json
import sys

# 导入所需的模块和类，准备与API进行交互
from fastchat.conversation import Conversation
from configs import TEMPERATURE
from http import HTTPStatus
from typing import List, Literal, Dict

from fastchat import conversation as conv
from server.model_workers.base import *
from server.model_workers.base import ApiEmbeddingsParams
from configs import logger, log_verbose


class QwenWorker(ApiModelWorker):
    """
    QwenWorker类，继承自ApiModelWorker，用于与Qwen模型进行交互。
    
    参数:
    - version: 模型版本，可选值为"qwen-turbo"或"qwen-plus"，默认为"qwen-turbo"。
    - model_names: 模型名称列表，默认为["qwen-api"]。
    - controller_addr: 控制器地址，默认为None。
    - worker_addr: 工作器地址，默认为None。
    
    属性:
    - version: 模型版本。
    """
    DEFAULT_EMBED_MODEL = "text-embedding-v1"

    def __init__(
        self,
        *,
        version: Literal["qwen-turbo", "qwen-plus"] = "qwen-turbo",
        model_names: List[str] = ["qwen-api"],
        controller_addr: str = None,
        worker_addr: str = None,
        **kwargs,
    ):
        # 初始化基类和本类特有的属性
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        与Qwen模型进行对话。
        
        参数:
        - params: 包含对话所需参数的对象，如模型名称、温度、API密钥和消息。
        
        返回:
        - 一个生成器，每次产生一个包含错误码、文本和可能的错误信息的字典。
        """
        import dashscope
        params.load_config(self.model_names[0])
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:params: {params}')

        gen = dashscope.Generation()
        responses = gen.call(
            model=params.version,
            temperature=params.temperature,
            api_key=params.api_key,
            messages=params.messages,
            result_format='message',  # 设置结果格式为消息格式。
            stream=True,
        )

        # 处理模型的响应，产生对话结果
        for resp in responses:
            if resp["status_code"] == 200:
                if choices := resp["output"]["choices"]:
                    yield {
                        "error_code": 0,
                        "text": choices[0]["message"]["content"],
                    }
            else:
                data = {
                    "error_code": resp["status_code"],
                    "text": resp["message"],
                    "error": {
                        "message": resp["message"],
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                }
                self.logger.error(f"请求千问 API 时发生错误：{data}")
                yield data

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        """
        计算文本嵌入。
        
        参数:
        - params: 包含嵌入所需参数的对象，如模型名称、API密钥和待嵌入的文本列表。
        
        返回:
        - 一个字典，包含嵌入向量或错误信息。
        """
        import dashscope
        params.load_config(self.model_names[0])
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:params: {params}')
        
        result = []
        i = 0
        # 分批处理文本，以不超过25行为限，获取嵌入向量
        while i < len(params.texts):
            texts = params.texts[i:i+25]
            resp = dashscope.TextEmbedding.call(
                model=params.embed_model or self.DEFAULT_EMBED_MODEL,
                input=texts, # 最大25行
                api_key=params.api_key,
            )
            if resp["status_code"] != 200:
                data = {
                            "code": resp["status_code"],
                            "msg": resp.message,
                            "error": {
                                "message": resp["message"],
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        }
                self.logger.error(f"请求千问 API 时发生错误：{data}")
                return data
            else:
                embeddings = [x["embedding"] for x in resp["output"]["embeddings"]]
                result += embeddings
            i += 25
        return {"code": 200, "data": result}

    def get_embeddings(self, params):
        # 该方法用于获取嵌入，但未具体实现
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。
        
        参数:
        - conv_template: 对话模板的字符串表示，默认为None。
        - model_path: 模型路径，默认为None。
        
        返回:
        - 一个Conversation实例，用于开始新的对话。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明、对人类有帮助的人工智能，你可以对人类提出的问题给出有用、详细、礼貌的回答。",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    # 用于本地离线测试FastAPI应用程序
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = QwenWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:20007",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=20007)