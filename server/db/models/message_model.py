from sqlalchemy import Column, Integer, String, DateTime, JSON, func

from server.db.base import Base


class MessageModel(Base):
    """
    聊天记录模型，用于表示用户和系统之间的聊天信息。
    
    属性:
    - id: 聊天记录的唯一标识符。
    - conversation_id: 对话框的唯一标识符，用于区分不同的对话。
    - chat_type: 聊天的类型，例如用户与客服的聊天、用户与机器人的聊天等。
    - query: 用户发起的问题或请求。
    - response: 系统针对用户问题给出的回复。
    - meta_data: 附加信息，用于存储与聊天相关的其他数据，如知识库ID等，方便后续扩展。
    - feedback_score: 用户对回复的评分，满分100分。
    - feedback_reason: 用户给出评分的理由。
    - create_time: 记录创建的时间。
    """

    __tablename__ = 'message'
    id = Column(String(32), primary_key=True, comment='聊天记录ID')
    conversation_id = Column(String(32), default=None, index=True, comment='对话框ID')
    chat_type = Column(String(50), comment='聊天类型')
    query = Column(String(4096), comment='用户问题')
    response = Column(String(4096), comment='模型回答')
    meta_data = Column(JSON, default={}, comment='附加信息，用于扩展')
    feedback_score = Column(Integer, default=-1, comment='用户评分')
    feedback_reason = Column(String(255), default="", comment='用户评分理由')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        # 返回表达式字符串，用于调试和日志记录
        return f"<message(id='{self.id}', conversation_id='{self.conversation_id}', chat_type='{self.chat_type}', query='{self.query}', response='{self.response}',meta_data='{self.meta_data}',feedback_score='{self.feedback_score}',feedback_reason='{self.feedback_reason}', create_time='{self.create_time}')>"