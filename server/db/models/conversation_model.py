from sqlalchemy import Column, Integer, String, DateTime, JSON, func
from server.db.base import Base


class ConversationModel(Base):
    """
    聊天记录模型，用于表示数据库中的会话信息。
    
    属性:
        id (Column[String[32]]): 对话框的唯一标识符，作为主键。
        name (Column[String[50]]): 对话框的名称。
        chat_type (Column[String[50]]): 聊天类型，如chat/agent_chat等。
        create_time (Column[DateTime]): 对话框的创建时间，默认为当前时间。
    """
    __tablename__ = 'conversation'  # 指定数据库表名为conversation
    id = Column(String(32), primary_key=True, comment='对话框ID')
    name = Column(String(50), comment='对话框名称')
    chat_type = Column(String(50), comment='聊天类型')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        # 返回表达式字符串，用于对象的可读性描述
        return f"<Conversation(id='{self.id}', name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"