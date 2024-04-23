from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from typing import List


def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    参数:
    - controller_address: Fastchat控制器的地址，如果未提供，则使用默认地址。
    - placeholder: 该参数未使用，仅为占位参数。
    返回值:
    - 返回一个包含模型名称及其配置信息的BaseResponse对象。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            models = r.json()["models"]
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")


def list_config_models(
    types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    从本地获取configs中配置的模型列表
    参数:
    - types: 模型配置项类别列表，例如"local", "online", "worker"。
    - placeholder: 该参数未使用，仅为占位参数。
    返回值:
    - 返回一个包含指定类型模型配置信息的BaseResponse对象。
    '''
    data = {}
    for type, models in list_config_llm_models().items():
        if type in types:
            data[type] = {m: get_model_config(m).data for m in models}
    return BaseResponse(data=data)


def get_model_config(
    model_name: str = Body(description="配置中LLM模型的名称"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    获取LLM模型配置项（合并后的）
    参数:
    - model_name: 配置中的LLM模型名称。
    - placeholder: 该参数未使用，仅为占位参数。
    返回值:
    - 返回一个包含指定模型配置信息的BaseResponse对象。
    '''
    config = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    for k, v in get_model_worker_config(model_name=model_name).items():
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    参数:
    - model_name: 要停止的LLM模型名称。
    - controller_address: Fastchat控制器的地址，如果未提供，则使用默认地址。
    返回值:
    - 返回一个表示操作结果的BaseResponse对象。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


def change_llm_model(
    model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
):
    '''
    向fastchat controller请求切换LLM模型。
    参数:
    - model_name: 当前运行的模型名称。
    - new_model_name: 要切换到的新模型名称。
    - controller_address: Fastchat控制器的地址，如果未提供，则使用默认地址。
    返回值:
    - 返回一个表示操作结果的BaseResponse对象。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # 等待新worker_model的时间
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")


def list_search_engines() -> BaseResponse:
    from server.chat.search_engine_chat import SEARCH_ENGINES

    '''
    获取可用的搜索引擎列表
    返回值:
    - 返回一个包含可用搜索引擎信息的BaseResponse对象。
    '''
    return BaseResponse(data=list(SEARCH_ENGINES))