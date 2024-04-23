from typing import List

from configs import (
    EMBEDDING_MODEL,
    KB_ROOT_PATH)

from abc import ABC, abstractmethod
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
import os
import shutil
from server.db.repository.knowledge_metadata_repository import add_summary_to_db, delete_summary_from_db

from langchain.docstore.document import Document


class KBSummaryService(ABC):
    """
    KBSummaryService类用于管理和操作知识库的摘要信息。
    属性:
    - kb_name: 知识库名称
    - embed_model: 嵌入模型名称
    - vs_path: 摘要向量存储路径
    - kb_path: 知识库路径

    方法:
    - __init__: 初始化知识库摘要服务
    - get_vs_path: 获取摘要向量存储路径
    - get_kb_path: 获取知识库路径
    - load_vector_store: 加载向量存储
    - add_kb_summary: 向知识库添加摘要
    - create_kb_summary: 创建知识库摘要
    - drop_kb_summary: 删除知识库摘要
    """

    kb_name: str
    embed_model: str
    vs_path: str
    kb_path: str

    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL
                 ):
        """
        初始化KBSummaryService实例。

        参数:
        - knowledge_base_name: 知识库名称
        - embed_model: 嵌入模型名称，默认为EMBEDDING_MODEL
        """
        self.kb_name = knowledge_base_name
        self.embed_model = embed_model

        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)


    def get_vs_path(self):
        """
        获取摘要向量存储路径。

        返回:
        - 摘要向量存储路径
        """
        return os.path.join(self.get_kb_path(), "summary_vector_store")

    def get_kb_path(self):
        """
        获取知识库路径。

        返回:
        - 知识库路径
        """
        return os.path.join(KB_ROOT_PATH, self.kb_name)

    def load_vector_store(self) -> ThreadSafeFaiss:
        """
        加载向量存储。

        返回:
        - 加载的向量存储实例
        """
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name="summary_vector_store",
                                               embed_model=self.embed_model,
                                               create=True)

    def add_kb_summary(self, summary_combine_docs: List[Document]):
        """
        向知识库添加摘要。

        参数:
        - summary_combine_docs: 文档列表，包含要添加的摘要

        返回:
        - 添加操作的状态
        """
        with self.load_vector_store().acquire() as vs:
            ids = vs.add_documents(documents=summary_combine_docs)
            vs.save_local(self.vs_path)

        summary_infos = [{"summary_context": doc.page_content,
                          "summary_id": id,
                          "doc_ids": doc.metadata.get('doc_ids'),
                          "metadata": doc.metadata} for id, doc in zip(ids, summary_combine_docs)]
        status = add_summary_to_db(kb_name=self.kb_name, summary_infos=summary_infos)
        return status

    def create_kb_summary(self):
        """
        创建知识库chunk摘要。
        """
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)

    def drop_kb_summary(self):
        """
        删除知识库chunk摘要。

        参数:
        - kb_name: 知识库名称

        返回:
        - 删除操作的状态
        """
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop(self.kb_name)
            shutil.rmtree(self.vs_path)
        delete_summary_from_db(kb_name=self.kb_name)