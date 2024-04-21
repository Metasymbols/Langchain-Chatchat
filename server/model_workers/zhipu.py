# 导入必要的包
from contextlib import contextmanager
import httpx
from fastchat.conversation import Conversation
from httpx_sse import EventSource
from server.model_workers.base import *
import sys
from typing import List, Dict, Iterator, Literal, Any
import jwt
import time

# 定义一个上下文管理器，用于连接并处理服务器发送的流数据
@contextmanager
def connect_sse(client: httpx.Client, method: str, url: str, **kwargs: Any):
    """
    连接到服务器发送的流数据事件源。
    
    :param client: httpx客户端，用于发起请求。
    :param method: HTTP请求方法。
    :param url: 请求的URL。
    :param kwargs: 传递给HTTP请求的额外参数。
    :yield: EventSource对象，用于处理服务器端的流数据事件。
    """
    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)

# 生成JWT Token
def generate_token(apikey: str, exp_seconds: int):
    """
    根据提供的API密钥生成JWT Token。
    
    :param apikey: API密钥，格式为"id.secret"。
    :param exp_seconds: Token的过期时间（秒）。
    :return: 生成的JWT Token字符串。
    """
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

# ChatGLMWorker类，用于与ChatGLM模型交互
class ChatGLMWorker(ApiModelWorker):
    DEFAULT_EMBED_MODEL = "embedding-2"
    
    def __init__(self, *, model_names: List[str] = ["zhipu-api"], controller_addr: str = None, worker_addr: str = None, version: Literal["glm-4"] = "glm-4", **kwargs):
        """
        初始化ChatGLMWorker实例。
        
        :param model_names: 模型名称列表。
        :param controller_addr: 控制器地址。
        :param worker_addr: 工作器地址。
        :param version: 模型版本。
        :param kwargs: 其他参数。
        """
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        super().__init__(**kwargs)
        self.version = version

    # 与ChatGLM模型进行对话
    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        """
        根据输入参数与ChatGLM模型进行对话。
        
        :param params: 对话参数，包括API密钥、消息等。
        :return: 生成器，每次返回对话的一条消息。
        """
        params.load_config(self.model_names[0])
        token = generate_token(params.api_key, 60)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "model": params.version,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": False
        }

        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        with httpx.Client(headers=headers) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            chunk = response.json()
            print(chunk)
            yield {"error_code": 0, "text": chunk["choices"][0]["message"]["content"]}

    # 计算嵌入向量
    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        """
        根据输入参数计算文本的嵌入向量。
        
        :param params: 嵌入参数，包括API密钥、待嵌入的文本等。
        :return: 包含嵌入向量结果的字典。
        """
        params.load_config(self.model_names[0])
        token = generate_token(params.api_key, 60)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        i = 0
        batch_size = 1
        result = []
        while i < len(params.texts):
            data = {
                "model": params.embed_model or self.DEFAULT_EMBED_MODEL,
                "input": "".join(params.texts[i: i + batch_size])
            }
            url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
            response = requests.post(url, headers=headers, json=data)
            ans = response.json()
            result.append(ans["data"][0]["embedding"])
            i += batch_size

        return {"code": 200, "data": result}
        
    # 获取嵌入向量（未实现）
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        根据提供的模板和模型路径创建对话模板。
        
        :param conv_template: 对话模板字符串。
        :param model_path: 模型路径。
        :return: Conversation对象。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是智谱AI小助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )

# 如果作为主模块运行，启动FastAPI服务
if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = ChatGLMWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21001",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21001)