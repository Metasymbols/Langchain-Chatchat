# ThreadSafeObject 类：提供线程安全的对象封装和管理。
class ThreadSafeObject:
    def __init__(self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None):
        """
        初始化一个线程安全的对象容器。

        :param key: 对象的唯一标识。
        :param obj: 存储的对象。
        :param pool: 对象所属的缓存池。
        """
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock()
        self._loaded = threading.Event()

    def __repr__(self) -> str:
        """
        返回对象的字符串表示。

        :return: 对象的字符串表示。
        """
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        """
        获取对象的标识符。

        :return: 对象的标识符。
        """
        return self._key

    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> FAISS:
        """
        上下文管理器，获取对象的锁并进行操作。

        :param owner: 操作的所有者标识。
        :param msg: 操作的消息。
        :yield: 对象的实例。
        """
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire()
            if self._pool is not None:
                self._pool._cache.move_to_end(self.key)
            if log_verbose:
                logger.info(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            if log_verbose:
                logger.info(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        """
        标记对象开始加载。
        """
        self._loaded.clear()

    def finish_loading(self):
        """
        标记对象加载完成。
        """
        self._loaded.set()

    def wait_for_loading(self):
        """
        等待对象加载完成。
        """
        self._loaded.wait()

    @property
    def obj(self):
        """
        获取对象属性。

        :return: 对象属性。
        """
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        """
        设置对象属性。

        :param val: 设置的值。
        """
        self._obj = val


# CachePool 类：缓存池，管理线程安全对象。
class CachePool:
    def __init__(self, cache_num: int = -1):
        """
        初始化缓存池。

        :param cache_num: 缓存的数量限制。
        """
        self._cache_num = cache_num
        self._cache = OrderedDict()
        self.atomic = threading.RLock()

    def keys(self) -> List[str]:
        """
        获取缓存池中的所有键。

        :return: 缓存池中的键列表。
        """
        return list(self._cache.keys())

    def _check_count(self):
        """
        检查缓存数量是否超过限制，如果超过则移除最早插入的项。
        """
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                self._cache.popitem(last=False)

    def get(self, key: str) -> ThreadSafeObject:
        """
        获取缓存中指定键的线程安全对象。

        :param key: 要获取的对象的键。
        :return: 线程安全对象。
        """
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            return cache

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        """
        在缓存中设置一个键值对。

        :param key: 对象的键。
        :param obj: 要设置的对象，必须是 ThreadSafeObject 的实例。
        :return: 设置的对象。
        """
        self._cache[key] = obj
        self._check_count()
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        """
        移除并返回缓存中指定键的线程安全对象。

        :param key: 要移除的对象的键。
        :return: 被移除的线程安全对象。
        """
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        """
        获取指定键的线程安全对象的锁并进行操作。

        :param key: 对象的键。
        :param owner: 操作的所有者标识。
        :param msg: 操作的消息。
        :return: 线程安全对象。
        """
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache

    def load_kb_embeddings(
            self,
            kb_name: str,
            embed_device: str = embedding_device(),
            default_embed_model: str = EMBEDDING_MODEL,
    ) -> Embeddings:
        """
        加载知识库嵌入模型。

        :param kb_name: 知识库名称。
        :param embed_device: 嵌入模型使用的设备。
        :param default_embed_model: 默认的嵌入模型。
        :return: 嵌入模型实例。
        """
        from server.db.repository.knowledge_base_repository import get_kb_detail
        from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter

        kb_detail = get_kb_detail(kb_name)
        embed_model = kb_detail.get("embed_model", default_embed_model)

        if embed_model in list_online_embed_models():
            return EmbeddingsFunAdapter(embed_model)
        else:
            return embeddings_pool.load_embeddings(model=embed_model, device=embed_device)


# EmbeddingsPool 类：嵌入池，专门用于管理嵌入模型的缓存。
class EmbeddingsPool(CachePool):
    def load_embeddings(self, model: str = None, device: str = None) -> Embeddings:
        """
        加载嵌入模型。

        :param model: 嵌入模型名称。
        :param device: 模型使用的设备。
        :return: 加载的嵌入模型实例。
        """
        self.atomic.acquire()
        model = model or EMBEDDING_MODEL
        device = embedding_device()
        key = (model, device)
        if not self.get(key):
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                # 根据模型名称选择合适的嵌入模型类进行初始化
                if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(model=model,
                                                  openai_api_key=get_model_path(model),
                                                  chunk_size=CHUNK_SIZE)
                elif 'bge-' in model:
                    from langchain.embeddings import HuggingFaceBgeEmbeddings
                    if 'zh' in model:
                        # for chinese model
                        query_instruction = "为这个句子生成表示以用于检索相关文章："
                    elif 'en' in model:
                        # for english model
                        query_instruction = "Represent this sentence for searching relevant passages:"
                    else:
                        # maybe ReRanker or else, just use empty string instead
                        query_instruction = ""
                    embeddings = HuggingFaceBgeEmbeddings(model_name=get_model_path(model),
                                                          model_kwargs={'device': device},
                                                          query_instruction=query_instruction)
                    if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
                        embeddings.query_instruction = ""
                else:
                    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name=get_model_path(model),
                                                       model_kwargs={'device': device})
                item.obj = embeddings
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(key).obj


# embeddings_pool 实例：嵌入池的单例实例。
embeddings_pool = EmbeddingsPool(cache_num=1)