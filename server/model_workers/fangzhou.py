from fastchat.conversation import Conversation
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
from typing import List, Literal, Dict
from configs import logger, log_verbose


class FangZhouWorker(ApiModelWorker):
    """
    火山方舟模型工作者类，用于与火山方舟API进行交互。
    """

    def __init__(self,
                 *,
                 model_names: List[str] = ["fangzhou-api"],  # 模型名称列表
                 controller_addr: str = None,  # 控制器地址
                 worker_addr: str = None,  # 工作者地址
                 version: Literal["chatglm-6b-model"] = "chatglm-6b-model",  # 模型版本
                 **kwargs,
                 ):
        """
        初始化FangZhouWorker。

        参数:
        - model_names: 模型名称列表，默认为["fangzhou-api"]。
        - controller_addr: 控制器地址，默认为None。
        - worker_addr: 工作者地址，默认为None。
        - version: 模型版本，默认为"chatglm-6b-model"。
        - **kwargs: 其他关键字参数。
        """
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        与火山方舟API进行聊天交互。

        参数:
        - params: ApiChatParams对象，包含聊天所需的配置参数。

        返回:
        - 与API交互的响应结果，为字典格式。
        """
        from volcengine.maas import MaasService

        params.load_config(self.model_names[0])
        maas = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
        maas.set_ak(params.api_key)
        maas.set_sk(params.secret_key)

        # 准备与火山方舟API交互的请求数据
        req = {
            "model": {
                "name": params.version,
            },
            "parameters": {
                # 这里的参数仅为示例，具体可用的参数请参考具体模型的 API 说明
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
            },
            "messages": params.messages,
        }

        text = ""
        if log_verbose:
            self.logger.info(f'{self.__class__.__name__}:maas: {maas}')
        for resp in maas.stream_chat(req):
            if error := resp.error:
                if error.code_n > 0:
                    # 处理API请求错误
                    data = {
                        "error_code": error.code_n,
                        "text": error.message,
                        "error": {
                            "message": error.message,
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                    self.logger.error(f"请求方舟 API 时发生错误：{data}")
                    yield data
                elif chunk := resp.choice.message.content:
                    text += chunk
                    yield {"error_code": 0, "text": text}
            else:
                # 处理未知错误
                data = {
                    "error_code": 500,
                    "text": f"请求方舟 API 时发生未知的错误: {resp}"
                }
                self.logger.error(data)
                yield data
                break

    def get_embeddings(self, params):
        """
        获取嵌入表示。

        参数:
        - params: 参数对象，具体取决于模型需求。

        说明:
        该方法目前仅打印信息，具体实现需根据需求完善。
        """
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。

        参数:
        - conv_template: 对话模板字符串，默认为None。
        - model_path: 模型路径字符串，默认为None。

        返回:
        - 初始化后的Conversation对象。
        """
        # 返回默认的对话模板
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明、对人类有帮助的人工智能，你可以对人类提出的问题给出有用、详细、礼貌的回答。",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = FangZhouWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21005",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21005)