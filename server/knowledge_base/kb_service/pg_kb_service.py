import json
from typing import List, Dict, Optional
import sqlalchemy.exc

from langchain.schema import Document
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from sqlalchemy import text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session

from configs import kbs_config

from server.knowledge_base.kb_service.base import SupportedVSType, KBService, EmbeddingsFunAdapter, \
    score_threshold_process
from server.knowledge_base.utils import KnowledgeFile
import shutil


class PGKBService(KBService):
    """
    PostgreSQL知识库服务类，提供对PostgreSQL数据库中存储的知识图谱数据的操作接口。
    """
    engine: Engine = sqlalchemy.create_engine(kbs_config.get("pg").get("connection_uri"), pool_size=10)

    def _load_pg_vector(self):
        """
        加载PostgreSQL向量存储。
        """
        self.pg_vector = PGVector(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                                  collection_name=self.kb_name,
                                  distance_strategy=DistanceStrategy.EUCLIDEAN,
                                  connection=PGKBService.engine,
                                  connection_string=kbs_config.get("pg").get("connection_uri"))

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据文档ID列表获取文档对象列表。

        :param ids: 文档ID列表。
        :return: 对应的文档对象列表。
        """
        if not ids:
            return []
        with Session(PGKBService.engine) as session:
            try:
                stmt = text("SELECT document, cmetadata FROM langchain_pg_embedding WHERE custom_id = ANY(:ids)")
                results = [Document(page_content=row[0], metadata=row[1]) for row in
                          session.execute(stmt, {'ids': ids}).fetchall()]
            except Exception as e:
                print(f"Error fetching documents by IDs: {e}")
                results = []
        return results

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """
        根据文档ID列表删除文档。

        :param ids: 文档ID列表。
        :return: 删除操作是否成功。
        """
        if not ids:
            return False
        try:
            return super().del_doc_by_ids(ids)
        except Exception as e:
            print(f"Error deleting documents by IDs: {e}")
            return False

    def do_init(self):
        """
        初始化向量存储。
        """
        self._load_pg_vector()

    def do_create_kb(self):
        """
        创建知识库。当前为空操作，子类可覆盖实现具体逻辑。
        """
        pass

    def vs_type(self) -> str:
        """
        获取向量存储类型。

        :return: 向量存储类型字符串。
        """
        return SupportedVSType.PG

    def execute_sql_safe(self, session, sql, params=None):
        """
        安全执行SQL语句。

        :param session: SQLAlchemy会话对象。
        :param sql: SQL语句字符串。
        :param params: SQL参数。
        :return: SQL执行结果。
        """
        try:
            result = session.execute(text(sql), params or {})
            session.commit()
            return result
        except sqlalchemy.exc.SQLAlchemyError as e:
            session.rollback()
            print(f"SQL execution error: {e}")
            return None

    def do_drop_kb(self):
        """
        删除知识库。包括从数据库中删除关联记录和物理删除存储路径。
        """
        with Session(PGKBService.engine) as session:
            collection_name = f"'{self.kb_name}'"
            # 删除 langchain_pg_embedding 表中关联到 langchain_pg_collection 表中的记录
            delete_embedding_sql = f'''
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                  SELECT uuid FROM langchain_pg_collection WHERE name = {collection_name}
                );
            '''
            # 删除 langchain_pg_collection 表中的记录
            delete_collection_sql = f'''
                DELETE FROM langchain_pg_collection WHERE name = {collection_name};
            '''
            self.execute_sql_safe(session, delete_embedding_sql)
            self.execute_sql_safe(session, delete_collection_sql)
            shutil.rmtree(self.kb_path, ignore_errors=True)

    def do_search(self, query: str, top_k: int, score_threshold: float):
        """
        根据查询文本进行搜索。

        :param query: 查询文本。
        :param top_k: 返回结果数量。
        :param score_threshold: 分数阈值。
        :return: 搜索结果列表。
        """
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        embeddings = embed_func.embed_query(query)
        docs = self.pg_vector.similarity_search_with_score_by_vector(embeddings, top_k)
        return score_threshold_process(score_threshold, top_k, docs)

    def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:
        """
        添加文档到知识库。

        :param docs: 文档对象列表。
        :return: 添加成功后的文档信息列表。
        """
        if not docs:
            return []
        ids = self.pg_vector.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        根据知识文件对象删除文档。

        :param kb_file: 知识文件对象。
        """
        with Session(PGKBService.engine) as session:
            filepath_replace = self.get_relative_source_path(kb_file.filepath)
            sql = '''
                DELETE FROM langchain_pg_embedding WHERE cmetadata::jsonb @> '{"source": "filepath"}'::jsonb;
            '''.replace("filepath", filepath_replace)
            self.execute_sql_safe(session, sql)

    def do_clear_vs(self):
        """
        清空向量存储，即删除所有集合及其数据。
        """
        self.pg_vector.delete_collection()
        self.pg_vector.create_collection()


if __name__ == '__main__':
    from server.db.base import Base, engine

    # Base.metadata.create_all(bind=engine)
    pGKBService = PGKBService("test")
    # pGKBService.create_kb()
    # pGKBService.add_doc(KnowledgeFile("README.md", "test"))
    # pGKBService.delete_doc(KnowledgeFile("README.md", "test"))
    # pGKBService.drop_kb()
    print(pGKBService.get_doc_by_ids(["f1e51390-3029-4a19-90dc-7118aaa25772"]))
    # print(pGKBService.search_docs("如何启动api服务"))