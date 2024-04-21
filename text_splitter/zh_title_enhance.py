from langchain.docstore.document import Document
import re

def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """
    检查文本中非字母字符的比例是否超过给定阈值。
    此功能有助于防止类似"-----------BREAK---------"的文本被标记为标题或叙述性文本。
    比例不计算空格。    
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except ZeroDivisionError:
        return False

def check_punctuation(text: str) -> bool:
    """
    检查文本是否以标点符号结尾。
    """
    ENDS_IN_PUNCT_PATTERN = r'[\.\,\?\!\，\。]$'
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    return ENDS_IN_PUNCT_RE.search(text) is not None

def is_possible_title(
        text: str,
        title_max_word_length: int = 20,
        non_alpha_threshold: float = 0.5,
) -> bool:
    """
    检查文本是否通过了有效标题的所有检查。
    """

    if len(text) == 0:
        print("Not a title. Text is empty.")
        return False

    if check_punctuation(text):
        return False

    if len(text) > title_max_word_length:
        return False

    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    if text.endswith((",", ".", "，", "。")):
        return False

    if text.isnumeric():
        print(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    if len(text) < 5:
        text_5 = text
    else:
        text_5 = text[:5]
    alpha_in_text_5 = sum(list(map(lambda x: x.isnumeric(), list(text_5))))
    if not alpha_in_text_5:
        return False

    return True


def zh_title_enhance(docs: Document, title_max_word_length: int = 20, non_alpha_threshold: float = 0.5) -> Document:
    """
    增强中文文档标题。
    
    遍历文档集合，将判断为可能的标题的页面内容设置为文档元数据中的类别为'cn_Title'，
    并且如果检测到标题，则在其他文档内容前添加与该标题相关的提示。
    
    参数
    ----------
    docs
        包含文档的集合
    title_max_word_length
        标题可以包含的最大单词数
    non_alpha_threshold
        文本被认为是标题所需的最少字母字符数

    返回
    -------
    Document
        处理后的文档集合
    """
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content, title_max_word_length, non_alpha_threshold):
                doc.metadata['category'] = 'cn_Title'
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")