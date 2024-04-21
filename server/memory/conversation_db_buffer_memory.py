import logging
from typing import Any, List, Dict

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from server.db.models.message_model import MessageModel


class ConversationBufferDBMemory(BaseChatMemory):
    """
    会话缓冲区数据库内存类，用于管理和存储会话历史记录。
    
    属性:
    conversation_id (str): 会话的唯一标识符。
    human_prefix (str): 人类消息的前缀，默认为"Human"。
    ai_prefix (str): AI消息的前缀，默认为"Assistant"。
    llm (BaseLanguageModel): 语言模型实例。
    memory_key (str): 内存关键字，默认为"history"。
    max_token_limit (int): 最大令牌限制，默认为2000。
    message_limit (int): 消息限制，默认为10。
    """
    
    conversation_id: str
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10

    @property
    def buffer(self) -> List[BaseMessage]:
        """
        获取会话缓冲区。
        
        返回:
        List[BaseMessage]: 会话历史记录的列表，包含人类消息和AI消息。
        """
        # 从数据库获取限定数量的消息，并反转时间顺序
        messages = filter_message(conversation_id=self.conversation_id, limit=self.message_limit)
        messages = list(reversed(messages))
        
        chat_messages: List[BaseMessage] = []
        for message in messages:
            chat_messages.append(HumanMessage(content=message["query"]))
            chat_messages.append(AIMessage(content=message["response"]))

        if not chat_messages:
            return []

        # 如果当前缓冲区长度超过最大令牌限制，则修剪消息
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))

        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """
        获取内存变量列表。
        
        返回:
        List[str]: 内存变量列表，此处固定为["history"]。
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载内存变量。
        
        参数:
        inputs (Dict[str, Any]): 输入参数字典。
        
        返回:
        Dict[str, Any]: 包含会话历史记录的字典。
        """
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        保存上下文信息。
        
        参数:
        inputs (Dict[str, Any]): 输入参数字典。
        outputs (Dict[str, str]): 输出结果字典。
        
        说明:
        该方法为空，不保存任何上下文信息。
        """
        pass

    def clear(self) -> None:
        """
        清除内存。
        
        说明:
        该方法为空，不执行任何清除操作。
        """
        pass