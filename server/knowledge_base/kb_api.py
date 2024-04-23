import urllib
from server.utils import BaseResponse, ListResponse
from server.knowledge_base.utils import validate_kb_name
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_base_repository import list_kbs_from_db
from configs import EMBEDDING_MODEL, logger, log_verbose
from fastapi import Body


def list_kbs():
    """
    获取知识库列表。

    返回:
        ListResponse: 包含知识库列表的响应对象。
    """
    return ListResponse(data=list_kbs_from_db())

def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    """
    创建指定的知识库。

    参数:
        knowledge_base_name (str): 知识库名称，必填项，用于指定要创建的知识库的名称。
        vector_store_type (str): 向量存储类型，默认为"faiss"，用于指定知识库的向量存储方式。
        embed_model (str): 嵌入模型，默认为EMBEDDING_MODEL，用于指定用于知识库的嵌入模型。

    返回:
        BaseResponse: 包含操作结果信息的响应对象，成功则返回状态码200，失败则返回其他状态码和错误信息。
    """
    # 验证知识库名称的合法性
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    # 检查知识库名称是否为空
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    # 检查是否存在同名知识库
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")

    # 创建知识库实例
    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"已新增知识库 {knowledge_base_name}")

def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    """
    删除指定的知识库。

    参数:
        knowledge_base_name (str): 知识库名称，必填项，用于指定要删除的知识库的名称。

    返回:
        BaseResponse: 包含操作结果信息的响应对象，成功则返回状态码200，失败则返回其他状态码和错误信息。
    """
    # 验证知识库名称的合法性
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    # 解码知识库名称
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    # 获取指定名称的知识库实例
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    # 检查知识库是否存在
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        # 尝试清除向量存储并删除知识库
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"成功删除知识库 {knowledge_base_name}")
    except Exception as e:
        msg = f"删除知识库时出现意外： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")