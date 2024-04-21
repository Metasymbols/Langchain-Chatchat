from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
try:
    from modelscope.pipelines import pipeline
except ImportError as e:
    raise ImportError(
        "Could not import modelscope python package. "
        "Please install modelscope with `pip install modelscope`. "
    ) from e


class AliTextSplitter(CharacterTextSplitter):
    """
    AliTextSplitter 类，继承自 CharacterTextSplitter，用于文本的分割。
    支持根据是否是PDF格式的文本，以及是否使用达摩院的语义分割模型来进行不同的文本分割。

    参数:
    - pdf: bool, 默认为 False。如果为 True，则表示输入的文本是 PDF 格式，需要进行特定的预处理。
    - model: str, 默认为 'damo/nlp_bert_document-segmentation_chinese-base'。达摩院语义分割模型的名称。
    - device: str, 默认为 'cpu'。指定模型运行的设备。
    - **kwargs: 传递给父类 CharacterTextSplitter 的额外关键字参数。
    """

    def __init__(self, pdf: bool = False, model='damo/nlp_bert_document-segmentation_chinese-base', 
                 device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.model = model
        self.device = device

    def preprocess_pdf_text(self, text: str) -> str:
        """对PDF文本进行预处理，减少换行符和空格"""
        pattern = re.compile(r"\n{3,}|^\s+$", re.MULTILINE)
        return pattern.sub("\n", text).strip()

    def segment_text(self, text: str) -> List[str]:
        """使用达摩院的语义分割模型进行文档分割"""
        try:
            p = pipeline(
                task="document-segmentation",
                model=self.model,
                device=self.device
            )
            result = p(documents=text)
            return [i for i in result["text"].split("\n\t") if i]
        except Exception as e:
            raise RuntimeError(
                "Failed to segment text with the specified model. "
                "Please check your model scope setup and internet connection."
            ) from e

    def split_text(self, text: str) -> List[str]:
        """
        分割给定的文本字符串 into a list of sentences or document segments.

        参数:
        - text: str, 需要分割的文本。

        返回值:
        - List[str], 分割后的文本片段列表。
        """
        if self.pdf:
            text = self.preprocess_pdf_text(text)
        
        return self.segment_text(text)