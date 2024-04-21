# 导入所需的模块和类
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func
from server.db.base import Base

class KnowledgeFileModel(Base):
    """
    知识文件模型，用于表示知识库中的文件信息。
    
    属性:
    - id: 知识文件ID，整数类型，主键，自增。
    - file_name: 文件名，字符串类型。
    - file_ext: 文件扩展名，字符串类型。
    - kb_name: 所属知识库名称，字符串类型。
    - document_loader_name: 文档加载器名称，字符串类型。
    - text_splitter_name: 文本分割器名称，字符串类型。
    - file_version: 文件版本，整数类型，默认为1。
    - file_mtime: 文件修改时间，浮点型，默认为0.0。
    - file_size: 文件大小，整数类型，默认为0。
    - custom_docs: 是否自定义docs，布尔类型，默认为False。
    - docs_count: 切分文档数量，整数类型，默认为0。
    - create_time: 创建时间，日期时间类型，默认为当前时间。
    """
    __tablename__ = 'knowledge_file'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识文件ID')
    file_name = Column(String(255), comment='文件名')
    file_ext = Column(String(10), comment='文件扩展名')
    kb_name = Column(String(50), comment='所属知识库名称')
    document_loader_name = Column(String(50), comment='文档加载器名称')
    text_splitter_name = Column(String(50), comment='文本分割器名称')
    file_version = Column(Integer, default=1, comment='文件版本')
    file_mtime = Column(Float, default=0.0, comment="文件修改时间")
    file_size = Column(Integer, default=0, comment="文件大小")
    custom_docs = Column(Boolean, default=False, comment="是否自定义docs")
    docs_count = Column(Integer, default=0, comment="切分文档数量")
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        # 返回模型对象的可读性好的字符串表示
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"

class FileDocModel(Base):
    """
    文件-向量库文档模型，用于表示文件和向量库中的文档之间的关系。
    
    属性:
    - id: ID，整数类型，主键，自增。
    - kb_name: 知识库名称，字符串类型。
    - file_name: 文件名称，字符串类型。
    - doc_id: 向量库文档ID，字符串类型。
    - meta_data: 元数据，JSON类型，默认为空字典。
    """
    __tablename__ = 'file_doc'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    kb_name = Column(String(50), comment='知识库名称')
    file_name = Column(String(255), comment='文件名称')
    doc_id = Column(String(50), comment="向量库文档ID")
    meta_data = Column(JSON, default={})

    def __repr__(self):
        # 返回模型对象的可读性好的字符串表示
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"