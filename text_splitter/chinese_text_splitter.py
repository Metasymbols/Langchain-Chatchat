from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class ChineseTextSplitter(CharacterTextSplitter):
    """
    用于分割中文文本的类，继承自CharacterTextSplitter。

    参数:
    - is_pdf: 布尔值，表示文本是否来自PDF。默认为False。
    - max_sentence_length: 整数，表示分割后的句子最大长度。默认为250。
    - **kwargs: 字典，传递给父类的额外参数。
    """

    def __init__(self, is_pdf: bool = False, max_sentence_length: int = 250, **kwargs):
        super().__init__(**kwargs)
        self.is_pdf = is_pdf
        self.max_sentence_length = max_sentence_length

    def clean_pdf_text(self, text: str) -> str:
        """
        清理PDF文本中的多余换行和空格。
        """
        # 合并清理多余换行和空格的正则表达式
        return re.sub(r'\n{3,}|[\s]+', ' ', text)

    def split_text1(self, text: str) -> List[str]:
        """
        分割文本为句子列表的另一种实现方法。

        参数:
        - text: 字符串，待分割的文本。

        返回值:
        - List[str]: 分割后的句子列表。
        """
        if self.is_pdf:
            text = self.clean_pdf_text(text)

        # 定义句子分隔符模式
        sentence_separator_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  
        sentences = []
        for part in sentence_separator_pattern.split(text):
            if sentence_separator_pattern.match(part) and sentences:
                sentences[-1] += part
            elif part:
                sentences.append(part)
        return sentences

    def split_text(self, text: str) -> List[str]:
        """
        分割传入的文本字符串为句子列表。

        参数:
        - text: 字符串，待分割的文本。

        返回值:
        - List[str]: 分割后的句子列表。
        """
        if self.is_pdf:
            text = self.clean_pdf_text(text)

        # 应用多个规则将文本按句子分隔
        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  
        text = re.sub(r'(\.{6})([^"’"])', r"\1\n\2", text)  # 优化了原正则表达式
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        text = text.rstrip()  

        # 去除分隔后产生的空字符串，并对过长的句子进行进一步分割
        split_text = [i for i in text.split("\n") if i]
        for sentence in split_text:
            if len(sentence) > self.max_sentence_length:
                # 对过长的句子进行多层次的分割处理
                first_level_split = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', sentence)
                first_level_parts = first_level_split.split("\n")
                for part in first_level_parts:
                    if len(part) > self.max_sentence_length:
                        second_level_split = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', part)
                        second_level_parts = second_level_split.split("\n")
                        for inner_part in second_level_parts:
                            if len(inner_part) > self.max_sentence_length:
                                third_level_split = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', inner_part)
                                second_level_index = second_level_parts.index(inner_part)
                                second_level_parts = second_level_parts[:second_level_index] + [i for i in third_level_split.split("\n") if i] + second_level_parts[
                                                                                                           second_level_index + 1:]
                        first_level_index = first_level_parts.index(part)
                        first_level_parts = first_level_parts[:first_level_index] + [i for i in second_level_parts if i] + first_level_parts[
                                                                                                       first_level_index + 1:]
                index = split_text.index(sentence)
                split_text = split_text[:index] + [i for i in first_level_parts if i] + split_text[index + 1:]
        return split_text