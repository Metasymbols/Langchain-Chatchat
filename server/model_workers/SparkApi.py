import base64
import datetime
import hashlib
import hmac
from urllib.parse import urlparse
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time


class Ws_Param(object):
    """
    Ws_Param 类用于生成WebSocket连接所需的参数和URL。

    参数:
    - APPID: 应用ID，用于标识应用程序。
    - APIKey: API密钥，用于认证。
    - APISecret: API密钥的秘密，用于生成签名。
    - Spark_url: WebSocket服务的URL。
    """

    # 初始化方法
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        # 解析URL并存储主机名和路径
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成用于WebSocket连接的URL
    def create_url(self):
        # 生成当前时间的时间戳，格式符合RFC1123
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 构建用于签名的原始字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 使用HMAC-SHA256算法对原始字符串进行签名
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        # 对签名结果进行Base64编码
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # 构建授权头部字符串，并进行Base64编码
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 准备请求的鉴权参数
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        
        # 生成带有鉴权参数的URL
        url = self.Spark_url + '?' + urlencode(v)
        
        # 返回生成的URL
        return url


def gen_params(appid, domain, question, temperature, max_token):
    """
    生成调用ChatAPI所需的参数。

    参数:
    - appid: 应用ID。
    - domain: 会话领域。
    - question: 用户的提问。
    - temperature: 生成回复的随机性温度。
    - max_token: 生成回复的最大令牌数。

    返回:
    - 用于调用ChatAPI的参数字典。
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"  # 用户ID，此处示例为固定值
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": 0.5,  # 随机响应阈值
                "max_tokens": max_token,  # 最大令牌数
                "auditing": "default",  # 审核设置
                "temperature": temperature,  # 生成回复的温度
            }
        },
        "payload": {
            "message": {
                "text": question  # 用户的提问
            }
        }
    }
    return data