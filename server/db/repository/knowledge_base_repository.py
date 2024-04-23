from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.session import with_session


# 将知识库添加到数据库中
@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    """
    向数据库中添加一个知识库实例。
    
    :param session: 数据库会话实例。
    :param kb_name: 知识库名称。
    :param kb_info: 知识库信息。
    :param vs_type: 知识库的版本类型。
    :param embed_model: 知识库的嵌入模型。
    :return: 布尔值，操作成功返回True。
    """
    # 查询是否已存在同名知识库
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if not kb:
        # 如果不存在，则创建新知识库实例并添加到会话
        kb = KnowledgeBaseModel(kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model)
        session.add(kb)
    else:  # 如果已存在，则更新知识库的信息
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    return True


# 从数据库中列出所有知识库名称
@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    """
    从数据库中列出文件数量大于指定值的所有知识库名称。
    
    :param session: 数据库会话实例。
    :param min_file_count: 最小文件数量，默认为-1表示不限制。
    :return: 知识库名称列表。
    """
    # 查询并返回知识库名称列表
    kbs = session.query(KnowledgeBaseModel.kb_name).filter(KnowledgeBaseModel.file_count > min_file_count).all()
    kbs = [kb[0] for kb in kbs]
    return kbs


# 检查知识库是否存在于数据库中
@with_session
def kb_exists(session, kb_name):
    """
    检查指定的知识库是否存在于数据库中。
    
    :param session: 数据库会话实例。
    :param kb_name: 知识库名称。
    :return: 布尔值，知识库存在返回True，否则返回False。
    """
    # 查询知识库实例是否存在
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    status = True if kb else False
    return status


# 从数据库加载知识库信息
@with_session
def load_kb_from_db(session, kb_name):
    """
    从数据库中加载指定知识库的信息。
    
    :param session: 数据库会话实例。
    :param kb_name: 知识库名称。
    :return: 知识库的名称、版本类型、嵌入模型元组，如果知识库不存在则返回None。
    """
    # 查询知识库信息
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


# 从数据库中删除知识库
@with_session
def delete_kb_from_db(session, kb_name):
    """
    从数据库中删除指定的知识库。
    
    :param session: 数据库会话实例。
    :param kb_name: 知识库名称。
    :return: 布尔值，操作成功返回True。
    """
    # 查询并删除知识库实例
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        session.delete(kb)
    return True


# 获取知识库的详细信息
@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    """
    获取指定知识库的详细信息。
    
    :param session: 数据库会话实例。
    :param kb_name: 知识库名称。
    :return: 包含知识库详细信息的字典，如果知识库不存在则返回空字典。
    """
    # 查询知识库详细信息
    kb: KnowledgeBaseModel = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}