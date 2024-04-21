async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[{
                                                             "role": "user",
                                                             "content": "我们来玩成语接龙，我先来，生龙活虎"
                                                         }, {
                                                             "role": "assistant",
                                                             "content": "虎头虎脑"
                                                         }]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ) -> EventSourceResponse:
    """
    与用户进行聊天的异步函数。
    
    参数:
    - query: 用户输入的查询字符串。
    - conversation_id: 对话框的唯一标识符。
    - history_len: 从数据库中读取的历史消息数量。
    - history: 历史对话，可以是一个整数（从数据库中读取指定数量的历史消息）或历史消息的列表。
    - stream: 是否以流式输出响应。
    - model_name: 使用的LLM模型名称。
    - temperature: LLM采样温度。
    - max_tokens: 限制LLM生成的Token数量。
    - prompt_name: 使用的Prompt模板名称。
    
    返回值:
    - EventSourceResponse: 一个服务器发送事件的响应，用于实时输出聊天结果。
    """

    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        
        # 设置消息存储和回调处理，为聊天过程做准备
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)

        # 调整最大Token数量的处理
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 根据配置初始化LLM模型
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        # 根据传入的历史消息或数据库中的历史消息准备聊天提示
        if history:  # 使用前端传入的历史消息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        elif conversation_id and history_len > 0:  # 从前端要求从数据库取历史消息
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:  # 没有提供历史消息
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        # 使用聊天提示和LLM模型创建对话链
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # 在后台启动一个任务来执行聊天过程
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        # 根据是否流式输出，以不同的方式返回聊天响应
        if stream:
            async for token in callback.aiter():
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)

        await task

    # 返回一个事件源响应，用于处理聊天交互
    return EventSourceResponse(chat_iterator())