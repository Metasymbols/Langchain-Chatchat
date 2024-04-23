from typing import List, Dict, Optional

from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus
import os

from configs import kbs_config
from server.db.repository import list_file_num_docs_id_by_kb_name_and_file_name

from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter, \
    score_threshold_process
from server.knowledge_base.utils import KnowledgeFile


class MilvusKBService(KBService):
    milvus: Milvus

    @staticmethod
    def get_collection(milvus_name):
        """
        获取Milvus集合。
        
        :param milvus_name: 集合名称。
        :return: 集合对象。
        """
        from pymilvus import Collection
        return Collection(milvus_name)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据ID列表获取文档。
        
        :param ids: 文档ID列表。
        :return: 文档列表。
        """
        result = []
        if self.milvus.col:
            data_list = self.milvus.col.query(expr=f'pk in {[int(_id) for _id in ids]}', output_fields=["*"])
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """
        根据ID列表删除文档。
        
        :param ids: 文档ID列表。
        """
        self.milvus.col.delete(expr=f'pk in {ids}')

    @staticmethod
    def search(milvus_name, content, limit=3):
        """
        在指定的Milvus集合中进行搜索。
        
        :param milvus_name: 集合名称。
        :param content: 搜索内容。
        :param limit: 返回结果的上限。
        :return: 搜索结果。
        """
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(content, "embeddings", search_params, limit=limit, output_fields=["content"])

    def do_create_kb(self):
        """
        创建知识库。当前为空实现，需根据实际需求进行扩展。
        """
        pass

    def vs_type(self) -> str:
        """
        获取向量存储类型。
        
        :return: 向量存储类型字符串。
        """
        return SupportedVSType.MILVUS

    def _load_milvus(self):
        """
        加载Milvus实例。
        """
        self.milvus = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                             collection_name=self.kb_name,
                             connection_args=kbs_config.get("milvus"),
                             index_params=kbs_config.get("milvus_kwargs")["index_params"],
                             search_params=kbs_config.get("milvus_kwargs")["search_params"]
                             )

    def do_init(self):
        """
        初始化Milvus连接。
        """
        self._load_milvus()

    def do_drop_kb(self):
        """
        删除知识库，释放并删除集合。
        """
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    def do_search(self, query: str, top_k: int, score_threshold: float):
        """
        执行搜索操作。
        
        :param query: 查询字符串。
        :param top_k: 返回结果数量。
        :param score_threshold: 分数阈值。
        :return: 搜索结果。
        """
        self._load_milvus()
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        embeddings = embed_func.embed_query(query)
        docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        return score_threshold_process(score_threshold, top_k, docs)

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        """
        添加文档到知识库。
        
        :param docs: 要添加的文档列表。
        :param kwargs: 其他参数。
        :return: 添加后的文档信息列表。
        """
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.milvus.fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.milvus._text_field, None)
            doc.metadata.pop(self.milvus._vector_field, None)

        ids = self.milvus.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        根据知识文件删除文档。
        
        :param kb_file: 知识文件对象。
        :param kwargs: 其他参数。
        """
        id_list = list_file_num_docs_id_by_kb_name_and_file_name(kb_file.kb_name, kb_file.filename)
        if self.milvus.col:
            self.milvus.col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self):
        """
        清除向量存储中的所有数据。
        """
        if self.milvus.col:
            self.do_drop_kb()
            self.do_init()


if __name__ == '__main__':
    # 测试建表使用
    from server.db.base import Base, engine

    Base.metadata.create_all(bind=engine)
    milvusService = MilvusKBService("test")
    # milvusService.add_doc(KnowledgeFile("README.md", "test"))

    print(milvusService.get_doc_by_ids(["444022434274215486"]))
    # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
    # milvusService.do_drop_kb()
    # print(milvusService.search_docs("如何启动api服务"))