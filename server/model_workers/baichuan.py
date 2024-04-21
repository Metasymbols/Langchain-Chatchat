import json
import time
import hashlib

# 导入对话管理类和基础模型工作类等必要模块
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import sys
import json
from typing import List, Literal, Dict
from configs import logger, log_verbose

def calculate_md5(input_string):
    """
    计算输入字符串的MD5加密值
    :param input_string: 需要加密的字符串
    :return: 加密后的字符串
    """
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    encrypted = md5.hexdigest()
    return encrypted


class BaiChuanWorker(ApiModelWorker):
    """
    百川模型工作类，用于与百川AI聊天模型交互
    """

    def __init__(
        self,
        *,
        controller_addr: str = None,
        worker_addr: str = None,
        model_names: List[str] = ["baichuan-api"],
        version: Literal["Baichuan2-53B"] = "Baichuan2-53B",
        **kwargs,
    ):
        """
        初始化百川模型工作器
        :param controller_addr: 控制器地址
        :param worker_addr: 工作器地址
        :param model_names: 模型名称列表
        :param version: 使用的模型版本
        :param kwargs: 其他参数
        """
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 32768)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        与百川AI模型进行对话
        :param params: 对话参数，包括模型名称、消息、温度等
        :return: 对话结果
        """
        params.load_config(self.model_names[0])

        # 构建请求URL和参数
        url = "https://api.baichuan-ai.com/v1/stream/chat"
        data = {
            "model": params.version,
            "messages": params.messages,
            "parameters": {"temperature": params.temperature}
        }

        # 生成请求签名
        json_data = json.dumps(data)
        time_stamp = int(time.time())
        signature = calculate_md5(params.secret_key + json_data + str(time_stamp))
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + params.api_key,
            "X-BC-Request-Id": "your requestId",
            "X-BC-Timestamp": str(time_stamp),
            "X-BC-Signature": signature,
            "X-BC-Sign-Algo": "MD5",
        }

        # 发送请求并处理响应
        text = ""
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:json_data: {json_data}')
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')

        with get_httpx_client() as client:
            with client.stream("POST", url, headers=headers, json=data) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    resp = json.loads(line)
                    if resp["code"] == 0:
                        text += resp["data"]["messages"][-1]["content"]
                        yield {
                            "error_code": resp["code"],
                            "text": text
                            }
                    else:
                        # 处理错误响应
                        data = {
                            "error_code": resp["code"],
                            "text": resp["msg"],
                            "error": {
                                "message": resp["msg"],
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        }
                        self.logger.error(f"请求百川 API 时发生错误：{data}")
                        yield data

    def get_embeddings(self, params):
        """
        获取嵌入表示（暂未实现）
        :param params: 参数
        """
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板
        :param conv_template: 对话模板字符串
        :param model_path: 模型路径
        :return: 对话对象
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    # 初始化并配置百川模型工作器
    worker = BaiChuanWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21007",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    # 启动API服务
    uvicorn.run(app, port=21007)
    # do_request()