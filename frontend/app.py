
import json

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001/api/v1/rag/query/stream"

st.set_page_config(
    page_title="ChatRAG",
    page_icon="💬",
    layout="centered",
)

st.title("💬 ChatRAG")
st.caption("基于混合检索与大语言模型的智能文档问答")

if "messages" not in st.session_state:
    st.session_state.messages = []

timings = {}

def stream_answer(question: str):
    """消费后端 SSE 事件，逐步返回生成的 token。"""
    sources = []
    retrieval = {}

    with requests.post(
        API_URL,
        json={"question": question},
        stream=True,
        timeout=(5, 300),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode("utf-8")

            # SSE 数据行格式：data: {...}
            if not line.startswith("data:"):
                continue

            payload = line[5:].strip()
            if not payload:
                continue

            event = json.loads(payload)
            event_type = event.get("type")

            if event_type == "sources":
                sources = event.get("sources", [])

            elif event_type == "token":
                content = event.get("content", "")
                if content:
                    yield {
                        "type": "token",
                        "content": content,
                    }
            elif event_type == "metrics":
                timings = event.get("timings", {})
                yield {
                    "type": "metrics",
                    "timings": timings,
                }

            elif event_type == "done":
                retrieval = event.get("retrieval", {})
                timings = event.get("timings", timings)

                yield {
                    "type": "done",
                    "sources": sources,
                    "retrieval": retrieval,
                    "timings": timings,
                }
                return

            elif event_type == "error":
                raise RuntimeError(
                    event.get("message", "后端流式生成失败")
                )

    # 兼容后端尚未发送 done 事件的情况
    yield {
        "type": "done",
        "sources": sources,
        "retrieval": retrieval,
        "timings": timings,
    }


# 展示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("查看参考来源"):
                st.json(message["sources"])

        if message.get("retrieval"):
            with st.expander("查看检索信息"):
                st.json(message["retrieval"])


question = st.chat_input("请输入你的问题……")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()

        answer = ""
        sources = []
        retrieval = {}

        try:
            status_placeholder.caption("正在检索文档并生成答案……")

            for event in stream_answer(question):
                if event["type"] == "token":
                    answer += event["content"]
                    answer_placeholder.markdown(answer + " ▌")

                elif event["type"] == "done":
                    sources = event.get("sources", [])
                    retrieval = event.get("retrieval", {})
                    timings = event.get("timings", {})

            answer_placeholder.markdown(answer or "模型没有返回答案。")
            status_placeholder.empty()

            if sources:
                with st.expander("查看参考来源"):
                    st.json(sources)

            if retrieval:
                with st.expander("查看检索信息"):
                    st.json(retrieval)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "retrieval": retrieval,
            })

        except requests.Timeout:
            status_placeholder.error(
                "请求超时，请检查后端状态或大模型 API 响应。"
            )

        except requests.RequestException as exc:
            status_placeholder.error(f"后端请求失败：{exc}")

        except (ValueError, TypeError, RuntimeError) as exc:
            status_placeholder.error(f"流式问答失败：{exc}")

if timings:
    st.subheader("⏱️ 性能统计")

    labels = {
        "retrieval_seconds": "混合检索",
        "context_seconds": "上下文构建",
        "first_token_seconds": "首 Token 延迟",
        "generation_seconds": "生成总耗时",
        "pipeline_seconds": "Pipeline 总耗时",
    }

    for key, label in labels.items():
        value = timings.get(key)
        if value is None:
            continue

        if isinstance(value, (int, float)):
            st.metric(label, f"{value:.3f} 秒")
        else:
            st.warning(
                f"{label}的数据类型异常："
                f"期望数字，实际为 {type(value).__name__}，值为 {value}"
            )
