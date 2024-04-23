import os
import sys

# 添加上级目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Any, List, Optional
from sentence_transformers import CrossEncoder
from typing import Optional, Sequence
from langchain_core.documents import Document
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from llama_index.bridge.pydantic import Field, PrivateAttr


class LangchainReranker(BaseDocumentCompressor):
    """
    使用Cohere Rerank API的文档压缩器。
    
    属性:
        model_name_or_path (str): 模型名称或路径。
        _model (Any): 模型实例。
        top_n (int): 返回结果的数量。
        device (str): 模型运行的设备（如"cuda"或"cpu"）。
        max_length (int): 输入的最大长度。
        batch_size (int): 批量大小。
        num_workers (int): 异步加载数据的线程数。
    """

    model_name_or_path: str = Field()
    _model: Any = PrivateAttr()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    num_workers: int = Field()

    def __init__(self,
                 model_name_or_path: str,
                 top_n: int = 3,
                 device: str = "cuda",
                 max_length: int = 1024,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 ):
        """
        初始化LangchainReranker。
        
        Args:
            model_name_or_path (str): 模型名称或路径。
            top_n (int): 返回结果的数量，默认为3。
            device (str): 模型运行的设备，默认为"cuda"。
            max_length (int): 输入的最大长度，默认为1024。
            batch_size (int): 批量大小，默认为32。
            num_workers (int): 异步加载数据的线程数，默认为0。
        """
        self._model = CrossEncoder(model_name=model_name_or_path, max_length=1024, device=device)
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        使用Cohere的rerank API压缩文档。
        
        Args:
            documents: 要压缩的文档序列。
            query: 用于压缩文档的查询字符串。
            callbacks: 在压缩过程中运行的回调。

        Returns:
            压缩后的文档序列。
        """
        if len(documents) == 0:  # 避免空的API调用
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        # 准备模型输入
        sentence_pairs = [[query, _doc] for _doc in _docs]
        # 使用模型进行预测
        results = self._model.predict(sentences=sentence_pairs,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      convert_to_tensor=True
                                      )
        top_k = self.top_n if self.top_n < len(results) else len(results)  # 确定返回结果的数量

        values, indices = results.topk(top_k)  # 获取top_k的结果
        final_results = []
        for value, index in zip(values, indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value  # 添加相关性分数到文档
            final_results.append(doc)
        return final_results


if __name__ == "__main__":
    # 从配置文件导入参数
    from configs import (LLM_MODELS,
                         VECTOR_SEARCH_TOP_K,
                         SCORE_THRESHOLD,
                         TEMPERATURE,
                         USE_RERANKER,
                         RERANKER_MODEL,
                         RERANKER_MAX_LENGTH,
                         MODEL_PATH)
    from server.utils import embedding_device

    # 根据配置初始化reranker模型
    if USE_RERANKER:
        reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL, "BAAI/bge-reranker-large")
        print("-----------------model path------------------")
        print(reranker_model_path)
        reranker_model = LangchainReranker(top_n=3,
                                           device=embedding_device(),
                                           max_length=RERANKER_MAX_LENGTH,
                                           model_name_or_path=reranker_model_path
                                           )