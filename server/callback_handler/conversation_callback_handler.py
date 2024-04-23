from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from server.db.repository import update_message

class ConversationCallbackHandler(BaseCallbackHandler):
    # 会话回调处理器类，用于处理对话中的回调事件
    raise_error: bool = True  # 是否在处理过程中抛出错误

    def __init__(self, conversation_id: str, message_id: str, chat_type: str, query: str):
        """
        初始化对话回调处理器
        
        :param conversation_id: 对话的唯一标识符
        :param message_id: 消息的唯一标识符
        :param chat_type: 对话的类型（如私聊、群聊等）
        :param query: 用户的查询或指令
        """
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.chat_type = chat_type
        self.query = query
        self.start_at = None  # 对话开始的时间戳

    @property
    def always_verbose(self) -> bool:
        """
        是否始终使用详细回调，即使在设置为非详细模式时也是如此
        
        :return: 始终返回True，表示始终使用详细回调
        """
        return True

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """
        当低层次模型（LLM）开始处理时的回调
        
        :param serialized: 序列化的模型输入数据
        :param prompts: 提示字符串列表，用于引导LLM的生成
        :param kwargs: 额外的关键字参数
        """
        # 此处可以添加处理逻辑，例如记录开始时间、存储prompts等
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        当低层次模型（LLM）处理结束时的回调
        
        :param response: 包含模型生成结果的对象
        :param kwargs: 额外的关键字参数
        """
        # 提取答案并更新消息内容
        answer = response.generations[0][0].text
        update_message(self.message_id, answer)  # 更新消息为生成的答案