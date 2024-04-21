from fastchat.conversation import Conversation
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
import json
from server.model_workers.base import ApiEmbeddingsParams
from server.utils import get_httpx_client
from typing import List, Dict
from configs import logger, log_verbose


class MiniMaxWorker(ApiModelWorker):
    """
    MiniMax模型工作者类，用于与MiniMax API进行交互。
    
    参数:
    - model_names: 模型名称列表，默认为["minimax-api"]。
    - controller_addr: 控制器地址，默认为None。
    - worker_addr: 工作者地址，默认为None。
    - version: 模型版本，默认为"abab5.5-chat"。
    - **kwargs: 其他关键字参数。
    
    属性:
    - version: 模型版本。
    """
    DEFAULT_EMBED_MODEL = "embo-01"

    def __init__(
        self,
        *,
        model_names: List[str] = ["minimax-api"],
        controller_addr: str = None,
        worker_addr: str = None,
        version: str = "abab5.5-chat",
        **kwargs,
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        super().__init__(**kwargs)
        self.version = version

    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        验证并转换消息格式。
        
        参数:
        - messages: 原始消息列表，每个消息为字典格式。
        
        返回: 转换后的消息列表，确保发送者类型和文本内容正确。
        """
        role_maps = {
            "USER": self.user_role,
            "assistant": self.ai_role,
            "system": "system",
        }
        messages = [{"sender_type": role_maps[x["role"]], "text": x["content"]} for x in messages]
        return messages

    def do_chat(self, params: ApiChatParams) -> Dict:
        """
        使用MiniMax API进行对话。
        
        参数:
        - params: ApiChatParams实例，包含对话所需的参数。
        
        返回: 与API交互的结果，通常为对话文本。
        """
        # 加载模型配置
        params.load_config(self.model_names[0])

        # 构建请求URL和头部
        url = 'https://api.minimax.chat/v1/text/chatcompletion{pro}?GroupId={group_id}'
        pro = "_pro" if params.is_pro else ""
        headers = {
            "Authorization": f"Bearer {params.api_key}",
            "Content-Type": "application/json",
        }
        messages = self.validate_messages(params.messages)
        data = {
            "model": params.version,
            "stream": True,
            "mask_sensitive_info": True,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "tokens_to_generate": params.max_tokens or 1024,
            # Minimax特有的参数，留空可能导致错误
            # "prompt": params.system_message or self.conv.system_message,
            # "bot_setting": [],
            # "role_meta": params.role_meta,
        }
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')
            logger.info(f'{self.__class__.__name__}:url: {url.format(pro=pro, group_id=params.group_id)}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')

        with get_httpx_client() as client:
            response = client.stream("POST",
                                    url.format(pro=pro, group_id=params.group_id),
                                    headers=headers,
                                    json=data)
            with response as r:
                text = ""
                for e in r.iter_text():
                    if not e.startswith("data: "):
                        # 错误处理
                        data = {
                                "error_code": 500,
                                "text": f"minimax返回错误的结果：{e}",
                                "error": {
                                    "message":  f"minimax返回错误的结果：{e}",
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": None,
                                }
                        }
                        self.logger.error(f"请求 MiniMax API 时发生错误：{data}")
                        yield data
                        continue

                    data = json.loads(e[6:])
                    if data.get("usage"):
                        break

                    if choices := data.get("choices"):
                        if chunk := choices[0].get("delta", ""):
                            text += chunk
                            yield {"error_code": 0, "text": text}

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        """
        使用MiniMax API获取嵌入向量。
        
        参数:
        - params: ApiEmbeddingsParams实例，包含嵌入所需的参数。
        
        返回: 嵌入向量结果。
        """
        # 加载模型配置
        params.load_config(self.model_names[0])
        url = f"https://api.minimax.chat/v1/embeddings?GroupId={params.group_id}"

        headers = {
            "Authorization": f"Bearer {params.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": params.embed_model or self.DEFAULT_EMBED_MODEL,
            "texts": [],
            "type": "query" if params.to_query else "db",
        }
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')

        with get_httpx_client() as client:
            result = []
            i = 0
            batch_size = 10
            while i < len(params.texts):
                texts = params.texts[i:i+batch_size]
                data["texts"] = texts
                r = client.post(url, headers=headers, json=data).json()
                if embeddings := r.get("vectors"):
                    result += embeddings
                elif error := r.get("base_resp"):
                    # 错误处理
                    data = {
                                "code": error["status_code"],
                                "msg": error["status_msg"],
                                "error": {
                                    "message":  error["status_msg"],
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": None,
                                }
                            }
                    self.logger.error(f"请求 MiniMax API 时发生错误：{data}")
                    return data
                i += batch_size
            return {"code": 200, "data": result}

    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        """
        创建对话模板。
        
        参数:
        - conv_template: 对话模板字符串，默认为None。
        - model_path: 模型路径，默认为None。
        
        返回: 初始化后的Conversation实例。
        """
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是MiniMax自主研发的大型语言模型，回答问题简洁有条理。",
            messages=[],
            roles=["USER", "BOT"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = MiniMaxWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21002",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21002)