from fastapi import Body, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from server.utils import (wrap_done, get_ChatOpenAI,
                        BaseResponse, get_prompt_template, get_temp_dir, run_in_thread_pool)
from server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.knowledge_base.utils import KnowledgeFile
import json
import os
from pathlib import Path


def _parse_files_in_thread(
    files: List[UploadFile],
    dir: str,
    zh_title_enhance: bool,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    通过多线程将上传的文件保存到对应目录内。
    生成器返回保存结果：[success or error, filename, msg, docs]
    
    :param files: 上传的文件列表
    :param dir: 文件保存的目录
    :param zh_title_enhance: 是否增强中文标题处理
    :param chunk_size: 文本块大小
    :param chunk_overlap: 文本块重叠大小
    """

    def parse_file(file: UploadFile) -> dict:
        '''
        保存单个文件。
        
        :param file: 单个上传文件
        :return: 保存结果，包括成功与否、文件名、消息和文档内容
        '''
        try:
            # 文件处理逻辑
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # 读取上传文件的内容
            
            # 确保文件保存目录存在
            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            return True, filename, f"成功上传文件 {filename}", docs
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            return False, filename, msg, []

    # 参数处理和多线程执行文件解析
    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result


def upload_temp_docs(
    files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
    prev_id: str = Form(None, description="前知识库ID"),
    chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    
    :param files: 上传的文件列表
    :param prev_id: 前一个知识库ID
    :param chunk_size: 文本块大小
    :param chunk_overlap: 文本块重叠大小
    :param zh_title_enhance: 是否增强中文标题处理
    :return: 保存结果，包括ID和失败的文件信息
    '''
    # 移除旧的向量库
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents = []
    path, id = get_temp_dir(prev_id)
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                        dir=path,
                                                        zh_title_enhance=zh_title_enhance,
                                                        chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})

    # 向量化处理文档
    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    return BaseResponse(data={"id": id, "failed_files": failed_files})


async def file_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                    knowledge_id: str = Body(..., description="临时知识库ID"),
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                    score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                    history: List[History] = Body([],
                                                description="历史对话",
                                                examples=[[{"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                            {"role": "assistant", "content": "虎头虎脑"}]]
                                                ),
                    stream: bool = Body(False, description="流式输出"),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                    prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                ):
    """
    基于文件知识库进行对话。
    
    :param query: 用户的查询输入
    :param knowledge_id: 临时知识库的ID
    :param top_k: 匹配向量的数量
    :param score_threshold: 知识库匹配相关度阈值
    :param history: 历史对话记录
    :param stream: 是否采用流式输出
    :param model_name: LLM模型名称
    :param temperature: LLM采样温度
    :param max_tokens: 限制LLM生成的Token数量
    :param prompt_name: 使用的Prompt模板名称
    :return: 对话响应
    """
    # 检查临时知识库是否存在
    if knowledge_id not in memo_faiss_pool.keys():
        return BaseResponse(code=404, msg=f"未找到临时知识库 {knowledge_id}，请先上传文件")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 模型和向量搜索设置
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        embed_func = EmbeddingsFunAdapter()
        embeddings = await embed_func.aembed_query(query)
        with memo_faiss_pool.acquire(knowledge_id) as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [x[0] for x in docs]

        # 构建对话上下文
        context = "\n".join([doc.page_content for doc in docs])
        if len(docs) == 0:  ## 如果没有找到相关文档，使用Empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        # 对话生成链式调用
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # 启动后台任务
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        # 根据是否流式输出处理对话结果
        if stream:
            async for token in callback.aiter():
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator())