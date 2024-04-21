# 导入相关模块和类
from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os
from langchain.schema import Document

# 为FAISS的InMemoryDocstore类打补丁，使其搜索功能返回文档ID
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc
InMemoryDocstore.search = _new_ds_search

class ThreadSafeFaiss(ThreadSafeObject):
    """
    提供一个线程安全的FAISS向量存储封装。
    """
    def __repr__(self) -> str:
        # 返回对象的字符串表示
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"


    def docs_count(self) -> int:
        # 返回向量库中的文档数量
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        # 将向量库保存到指定路径
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        # 清空向量库
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    """
    FAISS向量存储池，管理多个FAISS向量存储实例。
    """
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> FAISS:
        # 创建一个新的FAISS向量存储实例
        embeddings = EmbeddingsFunAdapter(embed_model)
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True, distance_strategy="METRIC_INNER_PRODUCT")
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str=None):
        # 保存指定的FAISS向量存储实例
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        # 卸载指定的FAISS向量存储实例
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    """
    用于管理基于知识库的FAISS向量存储池。
    """
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        # 加载或创建并返回一个指定的FAISS向量存储实例
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name))
        if cache is None:
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                vs_path = get_vs_path(kb_name, vector_name)

                if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    embeddings = self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)
                    vector_store = FAISS.load_local(vs_path, embeddings, normalize_L2=True, distance_strategy="METRIC_INNER_PRODUCT")
                elif create:
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                    vector_store.save_local(vs_path)
                else:
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    """
    用于管理内存中的FAISS向量存储池。
    """
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        # 加载或创建并返回一个指定的FAISS向量存储实例
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


# 初始化FAISS存储池
kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)



# 主程序入口
if __name__ == "__main__":
    import time, random
    from pprint import pprint

    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str, name: str):
        # 工作线程函数，用于演示向量存储的使用
        vs_name = "samples"
        time.sleep(random.randint(1, 5))
        embeddings = load_local_embeddings()
        r = random.randint(1, 3)

        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # 添加文档
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # 搜索文档
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # 删除文档
            logger.warning(f"清除 {vs_name} by {name}")
            kb_faiss_pool.get(vs_name).clear()

    threads = []
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()