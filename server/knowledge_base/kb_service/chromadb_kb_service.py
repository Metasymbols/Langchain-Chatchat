import uuid
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.api.types import (GetResult, QueryResult)
from langchain.docstore.document import Document

from configs import SCORE_THRESHOLD
from server.knowledge_base.kb_service.base import (EmbeddingsFunAdapter,
                                                   KBService, SupportedVSType)
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path

# 定义键名常量
DOCUMENTS_KEY = 'documents'
METADATAS_KEY = 'metadatas'
DISTANCES_KEY = 'distances'

def _get_result_to_documents(get_result: GetResult) -> List[Document]:
    """
    将查询结果转换为文档列表。

    :param get_result: 查询结果，包含文档内容和元数据。
    :return: 转换后的文档列表。
    """
    documents = get_result.get(DOCUMENTS_KEY, [])
    metadatas = get_result.get(METADATAS_KEY, [{}] * len(documents))
    
    document_list = []
    for page_content, metadata in zip(documents, metadatas):
        document_list.append(Document(**{'page_content': page_content, 'metadata': metadata}))

    return document_list


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    """
    将查询结果转换为文档和其对应分数的列表。

    :param results: 查询结果，包含文档、元数据和距离信息。
    :return: 转换后的文档和分数的元组列表。
    """
    documents = results.get(DOCUMENTS_KEY, [{}])[0]
    metadatas = results.get(METADATAS_KEY, [{}])[0]
    distances = results.get(DISTANCES_KEY, [0.0])[0]
    
    return [
        (Document(page_content=doc, metadata=metadata or {}), distance)
        for doc, metadata, distance in zip(documents, metadatas, distances)
    ]


class ChromaKBService(KBService):
    vs_path: str
    kb_path: str

    client = None
    collection = None

    def vs_type(self) -> str:
        """
        返回当前知识库服务使用的向量存储类型。

        :return: 向量存储类型字符串。
        """
        return SupportedVSType.CHROMADB

    def get_vs_path(self) -> str:
        """
        获取向量存储的路径。

        :return: 向量存储路径字符串。
        """
        return get_vs_path(self.kb_name, self.embed_model)

    def get_kb_path(self) -> str:
        """
        获取知识库的路径。

        :return: 知识库路径字符串。
        """
        return get_kb_path(self.kb_name)

    def do_init(self) -> None:
        """
        初始化Chroma向量存储客户端和知识库。
        """
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()
        self.client = chromadb.PersistentClient(path=self.vs_path)
        
        try:
            self.collection = self.client.get_or_create_collection(self.kb_name)
        except Exception as e:  # 更广泛的异常捕获
            # 应该记录日志或者处理异常
            raise e

    def do_create_kb(self) -> None:
        """
        创建知识库，实际上等同于创建一个集合。
        """
        self.collection = self.client.get_or_create_collection(self.kb_name)

    def do_drop_kb(self):
        """
        删除知识库，实际上等同于删除一个集合。
        """
        try:
            self.client.delete_collection(self.kb_name)
        except ValueError as e:
            if f"Collection {self.kb_name} does not exist." not in str(e):
                raise e

    def do_search(self, query: str, top_k: int, score_threshold: float = SCORE_THRESHOLD) -> List[Tuple[Document, float]]:
        """
        执行搜索操作。

        :param query: 查询字符串。
        :param top_k: 返回结果的数量。
        :param score_threshold: 分数阈值。
        :return: 匹配的文档和分数列表。
        """
        if not query:
            return []  # 针对空查询返回空列表
        
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        embeddings = embed_func.embed_query(query)
        query_result: QueryResult = self.collection.query(query_embeddings=embeddings, n_results=top_k)
        return _results_to_docs_and_scores(query_result)

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        """
        添加文档到知识库。

        :param docs: 要添加的文档列表。
        :return: 添加文档的信息列表。
        """
        if not docs:
            return []  # 针对空文档列表返回空列表
        
        doc_infos = []
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embeddings = embed_func.embed_documents(texts=texts)
        ids = [str(uuid.uuid1()) for _ in range(len(texts))]
        for _id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            self.collection.add(ids=_id, embeddings=embedding, metadatas=metadata, documents=text)
            doc_infos.append({"id": _id, "metadata": metadata})
        return doc_infos

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据ID获取文档。

        :param ids: 文档ID列表。
        :return: 对应的文档列表。
        """
        if not ids:
            return []  # 针对空ID列表返回空列表
        
        get_result: GetResult = self.collection.get(ids=ids)
        return _get_result_to_documents(get_result)

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """
        根据ID删除文档。

        :param ids: 要删除的文档ID列表。
        :return: 删除操作是否成功。
        """
        if not ids:
            return True  # 针对空ID列表认为删除操作成功
        
        self.collection.delete(ids=ids)
        return True

    def do_clear_vs(self):
        """
        清空向量存储，可能会等同于删除并重新创建集合。
        """
        self.do_drop_kb()

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        删除指定文件路径的文档。

        :param kb_file: 知识文件对象，包含要删除的文件路径。
        :return: 删除操作是否成功。
        """
        # 假设do_delete_doc的逻辑正确，但是需要确保安全性，这里不作修改
        return self.collection.delete(where={"source": kb_file.filepath})