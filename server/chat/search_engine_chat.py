from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from configs import (BING_SEARCH_URL, BING_SUBSCRIPTION_KEY, METAPHOR_API_KEY,
                     LLM_MODELS, SEARCH_ENGINE_TOP_K, TEMPERATURE, OVERLAP_SIZE)
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from sse_starlette import EventSourceResponse
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from server.chat.utils import History
from typing import AsyncIterable
import asyncio
import json
from typing import List, Optional, Dict
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from markdownify import markdownify


# 定义一个用于 Bing 搜索的函数
def bing_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    """
    使用Bing搜索引擎进行搜索。
    
    参数:
    - text: 要搜索的文本。
    - result_len: 搜索结果的数量，默认为SEARCH_ENGINE_TOP_K。
    - **kwargs: 传递给搜索API的其他参数。
    
    返回:
    - 搜索结果的列表，每个结果包括片段、标题和链接。
    """
    # 检查环境变量中是否设置了BING_SUBSCRIPTION_KEY和BING_SEARCH_URL
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        # 如果未设置，返回一个错误信息
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    # 使用Bing搜索API进行搜索
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)

# 定义一个用于 DuckDuckGo 搜索的函数
def duckduckgo_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    """
    使用DuckDuckGo搜索引擎进行搜索。
    
    参数:
    - text: 要搜索的文本。
    - result_len: 搜索结果的数量，默认为SEARCH_ENGINE_TOP_K。
    - **kwargs: 传递给搜索API的其他参数。
    
    返回:
    - 搜索结果的列表，每个结果包括片段、标题和链接。
    """
    # 使用DuckDuckGo搜索API进行搜索
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)

# 定义一个用于隐喻搜索的函数
def metaphor_search(
        text: str,
        result_len: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = OVERLAP_SIZE,
) -> List[Dict]:
    """
    使用Metaphor API进行搜索。
    
    参数:
    - text: 要搜索的文本。
    - result_len: 搜索结果的数量，默认为SEARCH_ENGINE_TOP_K。
    - split_result: 是否分割搜索结果，默认为False。
    - chunk_size: 分割搜索结果时的块大小，默认为500。
    - chunk_overlap: 分割搜索结果时的块重叠大小，默认为OVERLAP_SIZE。
    
    返回:
    - 搜索结果的列表，每个结果包括片段、链接和标题。
    """
    from metaphor_python import Metaphor

    # 检查环境变量中是否设置了METAPHOR_API_KEY
    if not METAPHOR_API_KEY:
        return []

    client = Metaphor(METAPHOR_API_KEY)
    search = client.search(text, num_results=result_len, use_autoprompt=True)
    contents = search.get_contents().contents
    for x in contents:
        x.extract = markdownify(x.extract)

    # 如果需要分割结果，则对Metaphor返回的长文本进行分词和检索
    if split_result:
        docs = [Document(page_content=x.extract,
                         metadata={"link": x.url, "title": x.title})
                for x in contents]
        text_splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", ".", " "],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splitted_docs = text_splitter.split_documents(docs)

        # 重新筛选出TOP_K个文档
        if len(splitted_docs) > result_len:
            normal = NormalizedLevenshtein()
            for x in splitted_docs:
                x.metadata["score"] = normal.similarity(text, x.page_content)
            splitted_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            splitted_docs = splitted_docs[:result_len]

        docs = [{"snippet": x.page_content,
                 "link": x.metadata["link"],
                 "title": x.metadata["title"]}
                for x in splitted_docs]
    else:
        docs = [{"snippet": x.extract,
                 "link": x.url,
                 "title": x.title}
                for x in contents]

    return docs


# 定义搜索引擎字典，映射搜索引擎名称到对应的搜索函数
SEARCH_ENGINES = {"bing": bing_search,
                  "duckduckgo": duckduckgo_search,
                  "metaphor": metaphor_search,
                  }


def search_result2docs(search_results):
    """
    将搜索结果转换为文档格式。
    
    参数:
    - search_results: 搜索结果的列表。
    
    返回:
    - 转换后的文档列表。
    """
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


# 异步查找搜索引擎结果
async def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
):
    """
    异步调用指定的搜索引擎进行搜索。
    
    参数:
    - query: 搜索查询字符串。
    - search_engine_name: 搜索引擎的名称。
    - top_k: 搜索结果的数量。
    - split_result: 是否分割搜索结果。
    
    返回:
    - 搜索结果的文档列表。
    """
    search_engine = SEARCH_ENGINES[search_engine_name]
    results = await run_in_threadpool(search_engine, query, result_len=top_k, split_result=split_result)
    docs = search_result2docs(results)
    return docs

# 用于聊天机器人的搜索功能
async def search_engine_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                             search_engine_name: str = Body(..., description="搜索引擎名称", examples=["duckduckgo"]),
                             top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                             history: List[History] = Body([],
                                                           description="历史对话",
                                                           examples=[[  # 示例对话
                                                               {"role": "user",
                                                                "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                               {"role": "assistant",
                                                                "content": "虎头虎脑"}]]
                                                           ),
                             stream: bool = Body(False, description="流式输出"),
                             model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                             temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                             max_tokens: Optional[int] = Body(None,
                                                              description="限制LLM生成Token数量，默认None代表模型最大值"),
                             prompt_name: str = Body("default",
                                                     description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                             split_result: bool = Body(False,
                                                       description="是否对搜索结果进行拆分（主要用于metaphor搜索引擎）")
                             ):
    """
    异步聊天功能，根据用户输入查询并使用指定的搜索引擎进行搜索，然后基于搜索结果和历史对话生成回复。
    
    参数:
    - query: 用户的查询字符串。
    - search_engine_name: 使用的搜索引擎名称。
    - top_k: 搜索结果的数量。
    - history: 历史对话列表。
    - stream: 是否启用流式输出。
    - model_name: LLM模型名称。
    - temperature: LLM采样温度。
    - max_tokens: 限制LLM生成的Token数量。
    - prompt_name: 使用的Prompt模板名称。
    - split_result: 是否对搜索结果进行拆分。
    
    返回:
    - 对用户查询的回复。
    """
    # 检查搜索引擎名称是否有效
    if search_engine_name not in SEARCH_ENGINES.keys():
        return BaseResponse(code=404, msg=f"未支持搜索引擎 {search_engine_name}")

    # 检查是否配置了BING_SUBSCRIPTION_KEY（如果使用Bing搜索引擎）
    if search_engine_name == "bing" and not BING_SUBSCRIPTION_KEY:
        return BaseResponse(code=404, msg=f"要使用Bing搜索引擎，需要设置 `BING_SUBSCRIPTION_KEY`")

    history = [History.from_data(h) for h in history]

    async def search_engine_chat_iterator(query: str,
                                          search_engine_name: str,
                                          top_k: int,
                                          history: Optional[List[History]],
                                          model_name: str = LLM_MODELS[0],
                                          prompt_name: str = prompt_name,) -> AsyncIterable[str]:
        
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        docs = await lookup_search_engine(query, search_engine_name, top_k, split_result=split_result)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("search_engine_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # 在后台开始一个任务
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = [
            f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(docs)
        ]

        if len(source_documents) == 0:  # 如果没有找到相关资料
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            async for token in callback.aiter():
                # 使用服务器发送事件流式传输响应
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

    return EventSourceResponse(search_engine_chat_iterator(query=query,
                                                           search_engine_name=search_engine_name,
                                                           top_k=top_k,
                                                           history=history,
                                                           model_name=model_name,
                                                           prompt_name=prompt_name),
                               )