import json
import asyncio

from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE, HISTORY_LEN, Agent_MODEL

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from typing import AsyncIterable, Optional, List

from server.utils import wrap_done, get_ChatOpenAI, get_prompt_template
from server.knowledge_base.kb_service.base import get_kb_details
from server.agent.custom_agent.ChatGLM3Agent import initialize_glm3_agent
from server.agent.tools_select import tools, tool_names
from server.agent.callbacks import CustomAsyncIteratorCallbackHandler, Status
from server.chat.utils import History
from server.agent import model_container
from server.agent.custom_template import CustomOutputParser, CustomPromptTemplate


""" 
 * 与代理进行聊天的异步函数。
 * 
 * @param query 用户输入的问题或对话内容。
 * @param history 过去的对话历史，用于上下文理解。
 * @param stream 是否以流式方式返回响应。
 * @param model_name 使用的LLM（大语言模型）的名称。
 * @param temperature LLM采样时的温度，影响生成结果的多样性。
 * @param max_tokens 限制LLM生成的token数量，None表示使用模型的最大值。
 * @param prompt_name 使用的prompt模板名称。
 * @return 返回一个异步可迭代的响应，如果是流式交互，则实时返回对话过程中的信息。
 """
async def agent_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     history: List[History] = Body([],
                                                   description="历史对话",
                                                   examples=[[  # 示例对话
                                                       {"role": "user", "content": "请使用知识库工具查询今天北京天气"},
                                                       {"role": "assistant",
                                                        "content": "使用天气查询工具查询到今天北京多云，10-14摄氏度，东北风2级，易感冒"}]]
                                                   ),
                     stream: bool = Body(False, description="流式输出"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     prompt_name: str = Body("default",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):
    history = [History.from_data(h) for h in history]

    """ *
     * 用于异步迭代生成聊天响应的内部函数。
     * 
     * @param query 用户输入的问题或对话内容。
     * @param history 过去的对话历史，用于上下文理解。
     * @param model_name 使用的LLM（大语言模型）的名称。
     * @param prompt_name 使用的prompt模板名称。
     * @return 返回一个异步可迭代的响应，包含对话过程中的信息。
     * """
    async def agent_chat_iterator(
            query: str,
            history: Optional[List[History]],
            model_name: str = LLM_MODELS[0],
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = CustomAsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 设置模型和回调
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        kb_list = {x["kb_name"]: x for x in get_kb_details()}  # 获取知识库列表
        model_container.DATABASE = {name: details['kb_info'] for name, details in kb_list.items()}  # 更新模型容器中的数据库

        if Agent_MODEL:  # 如果配置了代理模型
            model_agent = get_ChatOpenAI(
                model_name=Agent_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=[callback],
            )
            model_container.MODEL = model_agent
        else:
            model_container.MODEL = model

        # 设置prompt模板和输出解析器
        prompt_template = get_prompt_template("agent_chat", prompt_name)
        prompt_template_agent = CustomPromptTemplate(
            template=prompt_template,
            tools=tools,
            input_variables=["input", "intermediate_steps", "history"]
        )
        output_parser = CustomOutputParser()
        
        # 设置对话记忆和LLM链
        llm_chain = LLMChain(llm=model, prompt=prompt_template_agent)
        memory = ConversationBufferWindowMemory(k=HISTORY_LEN * 2)
        
        # 加载历史对话到记忆中
        for message in history:
            if message.role == 'user':
                memory.chat_memory.add_user_message(message.content)
            else:
                memory.chat_memory.add_ai_message(message.content)

        # 根据模型名称初始化对话执行器
        if "chatglm3" in model_container.MODEL.model_name or "zhipu-api" in model_container.MODEL.model_name:
            agent_executor = initialize_glm3_agent(
                llm=model,
                tools=tools,
                callback_manager=None,
                prompt=prompt_template,
                input_variables=["input", "intermediate_steps", "history"],
                memory=memory,
                verbose=True,
            )
        else:
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:", "Observation"],
                allowed_tools=tool_names,
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                                tools=tools,
                                                                verbose=True,
                                                                memory=memory,
                                                                )
        
        # 循环以处理对话
        while True:
            try:
                task = asyncio.create_task(wrap_done(
                    agent_executor.acall(query, callbacks=[callback], include_run_info=True),
                    callback.done))
                break
            except:
                pass

        # 如果是流式交互，则异步迭代并返回响应
        if stream:
            async for chunk in callback.aiter():
                tools_use = []
                # 解析并根据状态生成相应的聊天信息
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                elif data["status"] == Status.error:
                    tools_use.append("\n```\n")
                    tools_use.append("工具名称: " + data["tool_name"])
                    tools_use.append("工具状态: " + "调用失败")
                    tools_use.append("错误信息: " + data["error"])
                    tools_use.append("重新开始尝试")
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.tool_finish:
                    tools_use.append("\n```\n")
                    tools_use.append("工具名称: " + data["tool_name"])
                    tools_use.append("工具状态: " + "调用成功")
                    tools_use.append("工具输入: " + data["input_str"])
                    tools_use.append("工具输出: " + data["output_str"])
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.agent_finish:
                    yield json.dumps({"final_answer": data["final_answer"]}, ensure_ascii=False)
                else:
                    yield json.dumps({"answer": data["llm_token"]}, ensure_ascii=False)


        else:  # 非流式交互，等待整个对话完成
            answer = ""
            final_answer = ""
            async for chunk in callback.aiter():
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                if data["status"] == Status.error:
                    answer += "\n```\n"
                    answer += "工具名称: " + data["tool_name"] + "\n"
                    answer += "工具状态: " + "调用失败" + "\n"
                    answer += "错误信息: " + data["error"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.tool_finish:
                    answer += "\n```\n"
                    answer += "工具名称: " + data["tool_name"] + "\n"
                    answer += "工具状态: " + "调用成功" + "\n"
                    answer += "工具输入: " + data["input_str"] + "\n"
                    answer += "工具输出: " + data["output_str"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.agent_finish:
                    final_answer = data["final_answer"]
                else:
                    answer += data["llm_token"]

            yield json.dumps({"answer": answer, "final_answer": final_answer}, ensure_ascii=False)
        await task

    return EventSourceResponse(agent_chat_iterator(query=query,
                                                   history=history,
                                                   model_name=model_name,
                                                   prompt_name=prompt_name),
                               )