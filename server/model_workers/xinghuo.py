# 导入必要的模块和类
from fastchat.conversation import Conversation
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
import json
from server.model_workers import SparkApi
import websockets
from server.utils import iter_over_async, asyncio
from typing import List, Dict

# 定义一个异步函数，用于与Spark API进行WebSocket通信，获取生成的文本响应
async def request(appid, api_key, api_secret, Spark_url, domain, question, temperature, max_token):
    """
    异步发送WebSocket请求并获取响应的文本内容。

    参数:
    - appid: 应用ID
    - api_key: API密钥
    - api_secret: API密钥 secret
    - Spark_url: Spark API的URL
    - domain: 使用的领域模型
    - question: 提问的问题
    - temperature: 生成答案的随机性控制温度
    - max_token: 最大生成令牌数

    返回值:
    - 一个生成器，每次产生一条文本响应内容。
    """
    # 设置WebSocket连接参数并建立连接
    wsParam = SparkApi.Ws_Param(appid, api_key, api_secret, Spark_url)
    wsUrl = wsParam.create_url()
    data = SparkApi.gen_params(appid, domain, question, temperature, max_token)
    async with websockets.connect(wsUrl) as ws:
        await ws.send(json.dumps(data, ensure_ascii=False))
        finish = False
        # 循环接收并处理WebSocket响应，直到完成
        while not finish:
            chunk = await ws.recv()
            response = json.loads(chunk)
            if response.get("header", {}).get("status") == 2:
                finish = True
            if text := response.get("payload", {}).get("choices", {}).get("text"):
                yield text[0]["content"]

# 定义一个继承自ApiModelWorker的类，用于处理与兴火API的交互
class XingHuoWorker(ApiModelWorker):
    """
    与兴火（XingHuo）API交互的工作器类。

    参数:
    - model_names: 模型名称列表，默认为["xinghuo-api"]
    - controller_addr: 控制器地址，默认为None
    - worker_addr: 工作器地址，默认为None
    - version: API版本，默认为None
    """
    def __init__(  # 初始化函数
            self,
            *,
            model_names: List[str] = ["xinghuo-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            version: str = None,
            **kwargs,
    ):
        # 更新初始化参数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 8000)
        super().__init__(**kwargs)
        self.version = version

    # 处理聊天请求的方法
    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        处理聊天请求，与兴火API进行交互。

        参数:
        - params: 包含聊天请求信息的对象

        返回值:
        - 一个字典，包含与API交互的结果
        """
        params.load_config(self.model_names[0])

        # 定义不同版本API的配置映射
        version_mapping = {
            "v1.5": {"domain": "general", "url": "ws://spark-api.xf-yun.com/v1.1/chat", "max_tokens": 4000},
            "v2.0": {"domain": "generalv2", "url": "ws://spark-api.xf-yun.com/v2.1/chat", "max_tokens": 8000},
            "v3.0": {"domain": "generalv3", "url": "ws://spark-api.xf-yun.com/v3.1/chat", "max_tokens": 8000},
            "v3.5": {"domain": "generalv3", "url": "ws://spark-api.xf-yun.com/v3.5/chat", "max_tokens": 16000},
        }

        # 根据版本获取详细配置信息
        def get_version_details(version_key):
            return version_mapping.get(version_key, {"domain": None, "url": None})

        details = get_version_details(params.version)
        domain = details["domain"]
        Spark_url = details["url"]
        text = ""
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        params.max_tokens = min(details["max_tokens"], params.max_tokens or 0)
        # 使用异步循环遍历与API的交互结果，并生成响应
        for chunk in iter_over_async(
                request(params.APPID, params.api_key, params.APISecret, Spark_url, domain, params.messages,
                        params.temperature, params.max_tokens),
                loop=loop,
        ):
            if chunk:
                text += chunk
                yield {"error_code": 0, "text": text}

    # 获取嵌入函数（此处未实现，仅为示例）
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板的方法
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建一个对话模板。

        参数:
        - conv_template: 对话模板的字符串表示，默认为None
        - model_path: 模型路径，默认为None

        返回值:
        - 一个Conversation实例
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

# 如果直接运行此文件，启动FastAPI服务
if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = XingHuoWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21003",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21003)