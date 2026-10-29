# ChatRAG

面向本地 RAG 检索系统，支持 PDF 解析、章节拆分、文本分块、向量化、FAISS 索引构建与检索效果评测，并逐步扩展关键词检索和混合检索能力。

## 快速开始

### 步骤0：数据集

采用汽车行业国家标准文件作为数据集，数据可以从官方平台下载。

[全国标准信息公共服务平台](https://openstd.samr.gov.cn/bzgk/std/)



### 步骤1：准备Python环境
```aiignore
git clone https://github.com/hren-ron/ChatRAG.git
cd ChatRAG
conda create -n chat_rag python=3.10 -y
conda activate chat_rag
pip install -r requirements.txt

```

### 步骤2：配置模型
```aiignore
cp example.env .env
```
1. 在.env中配置云端模型信息（API_KEY, BASE_URL）；

2. 在config.py中配置本地数据集、模型、评测集等路径；

3. 将本地embedding、reranker模型下载到models目录下；

如果使用本地模型（如ollama）
1. ollama启动本地模型（qwen3.5:27b为例）：
```aiignore
ollama run qwen3.5:27b
```

2. 配置如下：
```aiignore
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
MODEL=qwen3.5:27b
```

### 步骤三：离线构建知识库

[项目使用流程（usage.md）](docs/usage.md)

### 步骤四：启动后台服务
```aiignore
python -m uvicorn src.api.main:app --port 8001
```

### 步骤五：启动前端服务
```aiignore
streamlit run frontend/app.py
```


## 核心功能

1. 文档解析：提取 PDF 文本，清理页眉、页脚等噪声，并按章节组织内容。

2. 文本分块：结合章节标题与文本长度生成检索 Chunk。

3. 向量检索：使用本地 BGE-M3 生成 Embedding，并通过 FAISS 检索。

4. 检索评测：使用统一评测集分析文档级与 Chunk 级召回效果。

4. 关键词与混合检索：逐步引入 BM25、RRF 等技术。


## 技术栈

Python、BGE-M3、Sentence Transformers、FAISS、Pydantic、BM25、jieba。

## 项目执行过程
```aiignore
汽车标准 PDF
    │
    ▼
PDF Parser
    │  提取文本、清理页眉页脚、恢复标题
    ▼
Chapter Splitter
    │  按一级/二级标题及附录生成章节文件
    ▼
ChunkLoader
    │  TXT → Structured Chunk
    ▼
SecondaryChunker
    │  按段落、句子和 Token 限制进行二次切分
    │  保留标题上下文、设置 overlap
    ▼
Retrieval Chunks
    ├──────────────────────────────┐
    ▼                              ▼
BGE-M3 Embedding                BM25 Tokenization
    │                              │
    ▼                              ▼
FAISS Vector Index             BM25 Inverted Index
    │                              │
    ▼                              ▼
Dense Retriever                BM25 Retriever
    └──────────────┬───────────────┘
                   ▼
              Retrieval Results
                   │
                   ▼
        Document-level / Chunk-level
               Evaluation
```
![项目执行过程](docs/images/framework.png)


## 目录结构参考
目录名以当前仓库实际代码为准，下面展示各模块的职责划分：

```aiignore
ChatRAG/
├── src/
│   ├── parser/             # PDF 解析、章节拆分、流程编排
│   ├── chunker/            # Chunk 模型、加载与二次切分
│   ├── models/              # BGE-M3 封装
│   ├── embedding/           # Embedding 流程
│   ├── vectorstore/         # FAISS 与 VectorStore
│   ├── retriever/           # Dense Retriever、BM25 Retriever、文本分析
│   └── evaluation/          # 指标、评测器、评测入口
├── data/
│   ├── pdf/                 # 原始 PDF
│   ├── output/              # 章节 TXT
│   ├── faiss/ 或 vectorstore/ # 索引和 chunks.json，按实际配置
│   ├── eval/                # 评测集与评测结果
│   └── dictionary/          # 汽车领域分词词典
├── Models/
│   └── bge-m3/              # 本地模型，不建议提交到 Git
├── requirements.txt
├── .gitignore
└── README.md
```

![目录结构参考](docs/images/dictionary.png)




## 后续计划
1. 中英文支持
2. OCR识别
3. RAG链路监控
4. RAG优化：embedding层优化（缓存，推理加速，top-K）、检索(元数据过滤)、rerank（小模型、微调）、LLM (上下文大小，stream、vLLM、缓存)
5. 架构层优化：文档预打分（高频问题、明确关键词） 、数据库分片、三级缓存、Hot data放内存
6. 在普通 RAG 流程稳定后，再扩展 Tool Calling 或 Multi-Agent
