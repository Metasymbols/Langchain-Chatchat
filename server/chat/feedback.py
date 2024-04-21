from fastapi import Body
from configs import logger, log_verbose
from server.utils import BaseResponse
from server.db.repository import feedback_message_to_db

def chat_feedback(message_id: str = Body("", max_length=32, description="聊天记录id"),
            score: int = Body(0, max=100, description="用户评分，满分100，越大表示评价越高"),
            reason: str = Body("", description="用户评分理由，比如不符合事实等")
            ):
    """
    向数据库记录聊天反馈信息。

    参数:
    - message_id (str): 聊天记录的唯一标识符，最大长度为32个字符。
    - score (int): 用户对聊天记录的评分，满分为100分，评分越高表示评价越好。
    - reason (str): 用户给出的评分理由，例如认为聊天记录不符合事实等。

    返回:
    - BaseResponse对象：包含操作结果的状态码和消息。成功时返回状态码200和成功消息，失败时返回状态码500和错误消息。
    """
    try:
        feedback_message_to_db(message_id, score, reason)
    except Exception as e:
        # 尝试将反馈信息存入数据库时遇到异常
        msg = f"反馈聊天记录出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    # 反馈信息成功存入数据库
    return BaseResponse(code=200, msg=f"已反馈聊天记录 {message_id}")