from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import BGE_M3_PATH, FAISS_PATH, RERANKER_PATH
from src.models.bge_m3_embedder import BGEEmbedder
from src.rag.context_build import ContextBuilder
from src.rag.generator import Generator
from src.rag.rag_pipeline import RAGPipeline
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.reranker import Reranker
from src.retriever.vector_retriever import VectorRetriever
from src.vector_store.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 RAG 组件，关闭时释放应用引用。"""
    print("Initializing RAG pipeline...")

    embedder = BGEEmbedder(model_path=BGE_M3_PATH)

    print("Loading FAISS...")
    vector_store = VectorStore.load(FAISS_PATH)

    print(f"Vector Count: {vector_store.size}")
    print(f"Embedding Dimension: {vector_store.faiss_store.dimension}")

    vector_retriever = VectorRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    bm25_retriever = BM25Retriever(
        chunks=vector_store.chunks,
    )

    reranker = Reranker(
        model_path=RERANKER_PATH,
        use_fp16=True,
        devices=["cuda:0"],
        normalize=True,
    )

    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        vector_retriever,
        reranker,
    )

    generator = Generator()
    context_builder = ContextBuilder()

    app.state.pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        context_builder=context_builder,
        generator=generator,
    )

    print("[STARTUP] RAG pipeline initialized.")
    print("[STARTUP] pipeline =", app.state.pipeline)

    yield

    # 应用关闭时清理引用；具体模型资源释放方式取决于各组件的实现。
    app.state.pipeline = None
    print("RAG service stopped.")


app = FastAPI(
    title="LocalRAG API",
    description="基于混合检索与大语言模型的标准文档问答服务",
    version="1.0.0",
    lifespan=lifespan,
)

print("当前加载的 API 文件：", __file__)
print("当前 FastAPI 应用：", id(app))



class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="用户提出的问题",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    retrieval: dict

@app.get("/health")
def health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)

    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "api_file": __file__,
    }

@app.get("/debug")
def debug(request: Request):
    return {
        "api_file": __file__,
        "app_id": id(request.app),
        "pipeline_ready": getattr(
            request.app.state, "pipeline", None
        ) is not None,
        "routes": [
            {
                "path": route.path,
                "name": route.name,
            }
            for route in request.app.routes
        ],
    }

@app.post("/api/v1/rag/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):

    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAGPipeline not be initialized",
        )

    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="question must not be null"
        )

    try:
        result = pipeline.query(question)
        return result
    except Exception as exc:
        # 服务端记录详细异常，客户端只返回通用错误。
        import logging
        logging.exception("RAG query failed")

        raise HTTPException(
            status_code=500,
            detail="问答处理失败，请稍后重试",
        ) from exc


print("========== 当前注册的路由 ==========")
for route in app.routes:
    print(route.path, getattr(route, "methods", None))
print("====================================")