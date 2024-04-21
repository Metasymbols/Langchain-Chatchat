from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Optional
import asyncio
from langchain.prompts import PromptTemplate

from server.utils import get_prompt_template


async def completion(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     stream: bool = Body(False, description="流式输出"),
                     echo: bool = Body(False, description="除了输出之外，还回显输入"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     prompt_name: str = Body("default",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):
    """
    异步完成LLM（大语言模型）的查询处理。支持流式输出和非流式输出两种模式。

    参数:
    - query: 用户输入的查询字符串。
    - stream: 是否以流式方式输出结果。
    - echo: 是否在输出结果的同时回显输入。
    - model_name: 使用的LLM模型名称。
    - temperature: LLM采样温度，控制生成结果的多样性。
    - max_tokens: 限制LLM生成的Token数量。
    - prompt_name: 使用的Prompt模板名称。

    返回值:
    - 根据stream参数，以流式或一次性返回的方式返回LLM的响应。
    """

    # 定义一个异步迭代器，用于处理LLM的查询响应。
    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        # 校验并设置max_tokens为合适的值。
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 初始化LLM模型和相关链式处理。
        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )

        # 根据prompt_name获取并设置prompt模板。
        prompt_template = get_prompt_template("completion", prompt_name)
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(prompt=prompt, llm=model)

        # 启动一个后台任务来处理模型的异步调用。
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        # 根据是否流式输出，以不同的方式处理并返回结果。
        if stream:
            async for token in callback.aiter():
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    # 返回一个事件源响应，用于处理流式输出。
    return EventSourceResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )