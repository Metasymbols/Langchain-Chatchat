from server.db.session import with_session
import uuid
from server.db.models.conversation_model import ConversationModel


@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    新增聊天记录到数据库。

    参数:
    - session: 数据库会话对象，用于执行数据库操作。
    - chat_type: 聊天类型，指定聊天记录的类型（如个人聊天或群组聊天）。
    - name: 聊天名称，可选参数，默认为空字符串。用于指定聊天的名称或主题。
    - conversation_id: 聊天记录的唯一标识符，可选参数，默认为None。如果未提供，则自动生成。

    返回值:
    - c.id: 新增聊天记录的唯一标识符。
    """
    # 如果没有提供conversation_id，则生成一个新的唯一标识符
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    # 创建一个新的ConversationModel实例，并初始化其属性
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    # 将新实例添加到数据库会话中，准备将其持久化
    session.add(c)
    # 返回新聊天记录的id
    return c.id