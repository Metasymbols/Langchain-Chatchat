from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger, log_verbose
from typing import List, Tuple, Dict, Union





class History(BaseModel):
    """
    对话历史信息类，用于表示一次对话的历史记录。

    可以通过字典或者元组的形式进行初始化和转化。

    属性:
    - role: str，对话角色，可以是"assistant"或"user"。
    - content: str，对话内容。
    """

    role: str = Field(..., regex=r"(assistant|user)")  # 增加正则表达式验证role的有效性
    content: str = Field(...)

    def to_msg_tuple(self) -> Tuple[str, str]:
        """
        将当前对话历史对象转化为（角色，内容）的元组形式。

        返回:
        - tuple，包含角色和内容的元组，角色可以是"ai"或"human"。
        """
        return ("ai" if self.role == "assistant" else "human"), self.content

    def to_msg_template(self, is_raw: bool = True) -> ChatMessagePromptTemplate:
        """
        将当前对话历史对象转化为聊天消息模板格式。

        参数:
        - is_raw: bool，标志内容是否为原始文本，默认为True。

        返回:
        - ChatMessagePromptTemplate对象，包含经过模板处理的角色和内容。
        """
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        # 根据is_raw标志，决定是否对内容进行原始标记处理
        # 使用格式化字符串优化
        content = f"{'{% raw %}'}{self.content}{'{% endraw %}'}" if is_raw else self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",  # 考虑到代码的可维护性，如果未来需要替换模板引擎，建议将其作为配置项
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List[str], Tuple[str, str], Dict[str, str]]) -> "History":
        """
        从不同的数据格式（列表、元组或字典）创建History对象。

        参数:
        - h: Union[List[str], Tuple[str, str], Dict[str, str]]，包含对话历史信息的数据结构。

        返回:
        - History对象，根据提供的数据初始化。
        """
        try:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                # 从列表或元组初始化
                return cls(role=h[0], content=h[1])
            elif isinstance(h, dict):
                # 从字典初始化
                return cls(**h)
            else:
                raise ValueError("输入数据格式不正确，应为列表、元组或字典。")
        except Exception as e:
            print(f"创建History对象失败: {str(e)}")
            # 根据实际情况决定是否需要抛出异常或返回None等
            raise
