# 导入所需的SQLAlchemy模块以及配置文件中的数据库URI
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
from configs import SQLALCHEMY_DATABASE_URI
import json

# 创建数据库引擎，同时配置自定义的JSON序列化器
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

# 配置一个局部会话，设置autocommit和autoflush为False，绑定到之前创建的数据库引擎
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明一个基于DeclarativeMeta的基类，用于构建ORM模型
Base: DeclarativeMeta = declarative_base()