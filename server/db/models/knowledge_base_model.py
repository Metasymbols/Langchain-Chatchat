from sqlalchemy import Column, Integer, String, DateTime, func
from server.db.base import Base

class KnowledgeBaseModel(Base):
    """
    知识库模型，用于表示知识库的数据结构。
    
    属性:
        id (Column(Integer)): 知识库的唯一标识符，自动递增。
        kb_name (Column(String(50))): 知识库的名称。
        kb_info (Column(String(200))): 知识库的简介，用于Agent展示。
        vs_type (Column(String(50))): 向量库的类型。
        embed_model (Column(String(50))): 嵌入模型的名称。
        file_count (Column(Integer)): 知识库中的文件数量，默认为0。
        create_time (Column(DateTime)): 知识库的创建时间，默认为当前时间。
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    kb_name = Column(String(50), comment='知识库名称')
    kb_info = Column(String(200), comment='知识库简介(用于Agent)')
    vs_type = Column(String(50), comment='向量库类型')
    embed_model = Column(String(50), comment='嵌入模型名称')
    file_count = Column(Integer, default=0, comment='文件数量')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        # 返回知识库模型的可读性好的字符串表示
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"