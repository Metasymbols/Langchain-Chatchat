from datetime import datetime
from sqlalchemy import Column, DateTime, String, Integer


class BaseModel:
    """
    基础模型类，作为所有数据库模型的基类。
    
    属性:
        id (Column): 主键ID，整数类型。
        create_time (Column): 创建时间，日期时间类型，默认为当前时间。
        update_time (Column): 更新时间，日期时间类型，默认为None，更新时自动设置为当前时间。
        create_by (Column): 创建者，字符串类型，默认为None。
        update_by (Column): 更新者，字符串类型，默认为None。
    """
    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    create_time = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="更新时间")
    create_by = Column(String, default=None, comment="创建者")
    update_by = Column(String, default=None, comment="更新者")