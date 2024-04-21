from functools import wraps
from contextlib import contextmanager
from server.db.base import SessionLocal
from sqlalchemy.orm import Session

@contextmanager
def session_scope() -> Session:
    """
    上下文管理器，用于自动获取 Session 并在操作完成后自动提交和关闭。
    这样可以确保在发生异常时自动回滚，避免数据不一致性。
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()  # 操作成功，提交事务
    except:
        session.rollback()  # 发生异常，回滚事务
        raise
    finally:
        session.close()  # 无论成功或失败，确保关闭 Session

def with_session(f):
    """
    装饰器，用于自动管理 Session 的生命周期。
    函数参数 f 必须接受一个 Session 作为其第一个参数。
    
    :param f: 需要进行数据库操作的函数，其第一个参数应为 Session
    :return: 包装后的函数，自动处理 Session 的获取、提交、关闭
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:  # 自动获取并进入 Session 上下文管理
            try:
                result = f(session, *args, **kwargs)  # 执行传入的函数
                session.commit()  # 操作成功，提交事务
                return result
            except:
                session.rollback()  # 发生异常，回滚事务
                raise

    return wrapper

def get_db() -> SessionLocal:
    """
    获取一个数据库会话实例。
    该函数使用 contextmanager 自动处理会话的开始和结束，但不会自动提交更改。
    
    :return: SessionLocal 实例
    """
    db = SessionLocal()
    try:
        yield db  # 提供给使用方操作数据库
    finally:
        db.close()  # 确保会话结束时关闭

def get_db0() -> SessionLocal:
    """
    直接返回一个数据库会话实例，不会自动关闭。
    使用方需要显式调用 close() 方法来关闭会话。
    
    :return: SessionLocal 实例
    """
    db = SessionLocal()
    return db