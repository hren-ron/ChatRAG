import os

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def load_pdf_text(pdf_path: str):
    """
    读取pdf的内容
    :param pdf_path:
    :return:
    """

    text = ""
    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def ask_llm(question: str, context: str):
    """
    直接向大模型提问
    :param question:
    :param context:
    :return:
    """

    client = OpenAI(
        api_key=os.getenv("BAILIAN_API_KEY"),
        base_url=os.getenv("BAILIAN_BASE_URL")
    )

    prompt = f"""
        请根据以下文档内容回答问题。如果文档中没有相关信息，请直接说不知道。
         
        文档内容：
        {context}
         
        问题：{question}
        回答："""

    response = client.chat.completions.create(
        model="qwen3.8-27b",
        messages=[{"role":"user", "content":prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content






if __name__=='__main__':

    pdf_path = "./产业图谱构建指南（浙江省）.pdf"

    print("load pdf text......")
    text = load_pdf_text(pdf_path)
    print(f"load finish, length={len(text)}")

    question = "这份文档里主要讲了什么？"
    print(f"\nquestion: {question}")
    print("thinking......")

    answer = ask_llm(question, text)
    print(f"\nanswer: {answer}")
