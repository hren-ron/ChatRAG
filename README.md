# ChatRAG

基于本地PDF文件构建智能问答系统

## 项目执行

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
1. 在.env中配置模型信息（API_KEY, BASE_URL）

2. 在config.py中配置本地Embedding模型路径，目前使用bge-m3模型。


## 功能模块

### 1. 文档解析

```PDFTextParser``` 负责从 PDF 中提取完整文本。

主要处理内容包括：

- 使用 PyMuPDF (fitz) 提取 PDF 文本；
根据文本的坐标信息恢复正确的阅读顺序；
- 根据页面中的空间位置信息恢复标题层级；
- 支持最多 4 级标题；
- 清理页眉、页脚和页码；
- 修复 PDF 文本提取过程中出现的标题断行问题；
- 修复中文、英文、数字之间的空格问题；
- 保留标准编号、章节编号等结构信息。

该过程最终返回一个完整的字符串：
```aiignore
text = parser.parse()
```

### 2. 文档切分

```ChapterSplitter``` 接收第一步产生的完整文本：
```aiignore
splitter = ChapterSplitter(text=text, output_dir=output_dir)
```
然后按照标题层级进行章节切分。

当前切分规则：

- 一级标题 → 创建新的 TXT 文件；
- 二级标题 → 创建新的 TXT 文件；
- 三级标题 → 不创建新文件，保留在当前章节中；
- 四级标题 → 不创建新文件，保留在当前章节中；
- 附录 A、附录 B 等附录 → 创建新的 TXT 文件。

三级和四级标题不会被丢弃，而是作为当前章节的上下文保留下来。


```PDFPipeline``` 不负责具体的 PDF 解析和章节切分逻辑，只负责将两个独立过程串联起来。

执行流程：
```aiignore
parser = PDFTextParser(pdf_path=pdf_path)

text = parser.parse()

splitter = ChapterSplitter(text=text,output_dir=output_dir)

splitter.split()
```

因此整个 Pipeline 为：

```aiignore
PDF
 │
 ▼
PDFTextParser
 │
 │ 完整文本（内存）
 ▼
ChapterSplitter
 │
 ▼
多个章节TXT
```

### 3.Chunker
Chunker 负责将已经按照章节划分好的 TXT 文档，进一步处理为适合 RAG 检索的 Retrieval Chunk。

整体流程如下：

```aiignore
章节 TXT
   │
   ▼
ChunkLoader
   │
   ▼
Structured Chunk
   │
   ▼
SecondaryChunker
   │
   ├── 标题层级识别
   ├── 上下文构建
   ├── 段落切分
   ├── 句子切分
   ├── Token 切分
   └── Overlap
   │
   ▼
Retrieval Chunk
   │
   ├── Embedding
   ├── BM25
   └── Vector DB
```

#### 3.1 ChunkLoader
ChunkLoader 负责读取已经生成好的章节 TXT 文件，并转换成统一的 Chunk 对象。

输入目录：
```aiignore
data/output/
├── GB+1589-2026/
│   ├── 001_1_范围.txt
│   ├── 002_2_规范性引用文件.txt
│   ├── 003_3_术语和定义.txt
│   └── ...
│
└── GB+7258-2017/
    ├── 001_1_范围.txt
    └── ...
```
例如：
```
004_4_技术要求.txt
```
会被加载成一个 Structured Chunk。


#### 3.2 Structured Chunk

Structured Chunk 保留原始章节的完整内容。

例如：
```aiignore
4 技术要求

4.1 制动系统

4.1.1 制动性能

车辆的制动性能应满足……

4.1.2 制动距离

车辆的制动距离应满足……

4.2 转向系统

……
```
一个章节对应一个 Structured Chunk。

Structured Chunk 的特点：
- 保留完整章节内容
- 保留标题层级
- 适合保存原始文档结构
- 不一定适合直接进行 Embedding

因为某些章节可能包含数千甚至上万个 Token。

#### 3.3 SecondaryChunker
SecondaryChunker 负责将 Structured Chunk 进一步切分成适合检索的 Retrieval Chunk。

核心目标：
```aiignore
Structured Chunk
        │
        ▼
多个 Retrieval Chunk
```
例如：
```
4 技术要求
```
可能最终变成：
```
gb_1589_2026_004_4_001
gb_1589_2026_004_4_002
gb_1589_2026_004_4_003
```


##### 3.3.1 二次切分策略
SecondaryChunker 按以下顺序进行切分：
```
段落
 ↓
句子
 ↓
Token
```
1. 优先按照段落进行切分。如果当前 Chunk 还能容纳整个段落，则直接加入。
2. 如果一个段落本身超过 Chunk 最大 Token 数，则进一步按照句子切分。
3. 如果单个句子仍然超过 Token 限制，则最终按照 Token 进行切分。

因此整体策略是： 优先保持语义完整

##### 3.3.2 Retrieval Chunk
最终生成的 Retrieval Chunk 包含：
```aiignore
Chunk 
├── chunk_id 
├── document_id 
├── source 
├── version 
├── chapter 
├── title 
├── level 
├── parent_path 
├── content 
├── content_hash 
├── chunk_strategy 
├── chunk_index 
├── token_count 
└── ...
```

### 注意
1. 后续使用若需进一步提升效果，可以根据文档格式修改文档解析方法。

### todo
1. 中英文支持
2. OCR识别
3. RAG链路监控
4. RAG优化：embedding层优化（缓存，推理加速，top-K）、检索(元数据过滤)、rerank（小模型、微调）、LLM (上下文大小，stream、vLLM、缓存)
5. 架构层优化：文档预打分（高频问题、明确关键词） 、数据库分片、三级缓存、Hot data放内存
