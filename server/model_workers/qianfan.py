# 导入模块
import sys
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from cachetools import cached, TTLCache
import json
from fastchat import conversation as conv
import sys
from server.model_workers.base import ApiEmbeddingsParams
from typing import List, Literal, Dict
from configs import logger, log_verbose

# 定义支持的模型版本及其对应API路径
MODEL_VERSIONS = {
    "ernie-bot-4": "completions_pro",
    "ernie-bot": "completions",
    "ernie-bot-turbo": "eb-instant",
    "bloomz-7b": "bloomz_7b1",
    "qianfan-bloomz-7b-c": "qianfan_bloomz_7b_compressed",
    "llama2-7b-chat": "llama_2_7b",
    "llama2-13b-chat": "llama_2_13b",
    "llama2-70b-chat": "llama_2_70b",
    "qianfan-llama2-ch-7b": "qianfan_chinese_llama_2_7b",
    "chatglm2-6b-32k": "chatglm2_6b_32k",
    "aquilachat-7b": "aquilachat_7b",
    # 以下模型暂未发布
    # "linly-llama2-ch-7b": "", 
    # "linly-llama2-ch-13b": "", 
    # "chatglm2-6b": "", 
    # "chatglm2-6b-int4": "", 
    # "falcon-7b": "", 
    # "falcon-180b-chat": "", 
    # "falcon-40b": "", 
    # "rwkv4-world": "", 
    # "rwkv5-world": "", 
    # "rwkv4-pile-14b": "", 
    # "rwkv4-raven-14b": "", 
    # "open-llama-7b": "", 
    # "dolly-12b": "", 
    # "mpt-7b-instruct": "", 
    # "mpt-30b-instruct": "", 
    # "OA-Pythia-12B-SFT-4": "", 
    # "xverse-13b": "", 
    # 以下为企业测试，需要单独申请
    # "flan-ul2": "",
    # "Cerebras-GPT-6.7B": ""
    # "Pythia-6.9B": ""
}

@cached(TTLCache(1, 1800))  # 使用缓存来存储百度访问令牌，每30分钟刷新一次
def get_baidu_access_token(api_key: str, secret_key: str) -> str:
    """
    使用提供的API密钥和密钥生成百度鉴权签名（Access Token）。
    :param api_key: 百度API的密钥（API Key）。
    :param secret_key: 百度API的密钥（Secret Key）。
    :return: 返回获取到的访问令牌（Access Token），如果获取失败则返回None。
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    try:
        with get_httpx_client() as client:
            return client.get(url, params=params).json().get("access_token")
    except Exception as e:
        print(f"获取百度访问令牌失败: {e}")

class QianFanWorker(ApiModelWorker):
    """
    百度千帆模型工作的类，支持聊天和嵌入式功能。
    """
    DEFAULT_EMBED_MODEL = "embedding-v1"

    def __init__(  # 初始化函数，配置千帆模型的工作参数
            self,
            *,
            version: Literal["ernie-bot", "ernie-bot-turbo"] = "ernie-bot",
            model_names: List[str] = ["qianfan-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            **kwargs,
    ):
        # 更新初始化参数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        # 调用基类的初始化函数
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        使用百度千帆模型进行聊天。
        :param params: 包含聊天所需参数的对象，例如消息、温度等。
        :return: 生成的聊天回复，以字典格式包含错误码和文本。
        """
        # 加载聊天模型配置
        params.load_config(self.model_names[0])
        # 构建请求的基础URL，包含访问令牌和模型版本
        BASE_URL = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat' \
                   '/{model_version}?access_token={access_token}'

        # 获取访问令牌
        access_token = get_baidu_access_token(params.api_key, params.secret_key)
        if not access_token:
            # 如果获取访问令牌失败，返回错误信息
            yield {
                "error_code": 403,
                "text": f"获取访问令牌失败。API密钥和密钥 secret key 配置正确吗？",
            }

        # 构建完整的聊天请求URL和参数
        url = BASE_URL.format(
            model_version=params.version_url or MODEL_VERSIONS[params.version.lower()],
            access_token=access_token,
        )
        payload = {
            "messages": params.messages,
            "temperature": params.temperature,
            "stream": True
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        # 准备日志记录
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {payload}')
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')

        # 发送聊天请求并处理响应
        with get_httpx_client() as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    resp = json.loads(line)

                    # 处理正常响应或错误响应
                    if "result" in resp.keys():
                        text += resp["result"]
                        yield {
                            "error_code": 0,
                            "text": text
                        }
                    else:
                        data = {
                            "error_code": resp["error_code"],
                            "text": resp["error_msg"],
                            "error": {
                                "message": resp["error_msg"],
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        }
                        self.logger.error(f"请求千帆 API 时发生错误：{data}")
                        yield data

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        """
        使用百度千帆进行文本嵌入处理。
        :param params: 包含嵌入处理所需参数的对象，例如文本、嵌入模型等。
        :return: 返回嵌入结果，以字典格式包含错误码、消息和嵌入数据。
        """
        # 加载嵌入模型配置
        params.load_config(self.model_names[0])
        embed_model = params.embed_model or self.DEFAULT_EMBED_MODEL
        access_token = get_baidu_access_token(params.api_key, params.secret_key)
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{embed_model}?access_token={access_token}"
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:url: {url}')

        # 批量处理文本嵌入
        with get_httpx_client() as client:
            result = []
            i = 0
            batch_size = 10
            while i < len(params.texts):
                texts = params.texts[i:i + batch_size]
                resp = client.post(url, json={"input": texts}).json()
                if "error_code" in resp:
                    # 处理错误响应
                    data = {
                        "code": resp["error_code"],
                        "msg": resp["error_msg"],
                        "error": {
                            "message": resp["error_msg"],
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                    self.logger.error(f"请求千帆 API 时发生错误：{data}")
                    return data
                else:
                    # 处理正常响应，收集嵌入数据
                    embeddings = [x["embedding"] for x in resp.get("data", [])]
                    result += embeddings
                i += batch_size
            return {"code": 200, "data": result}

    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。
        :return: 返回一个对话对象。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    # 初始化QianFanWorker
    worker = QianFanWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21004"
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 创建并运行FastAPI应用
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21004)