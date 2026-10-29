
## 整体流程
![整体流程](images/rag.PNG)

### 1. 数据准备
1. 检查本地模型目录(**models**)
2. 确认本地 PDF 已放入 **data/pdf/** 目录

### 2. PDF 解析与章节拆分

运行 PDF 处理流程：
```aiignore
python -m src.parser.run_pipeline
```

该流程由两个模块协作完成：

1. src/parser/pdf_parser.py：提取 PDF 文本，清理页眉、页脚和页码等噪声。

2. src/parser/chapter_splitter.py：根据标题层级拆分章节，生成章节 TXT 文件。

检查输出目录：
```aiignore
data/txt/
```

检查生成的章节文本，确认标准编号、条款标题、正文、附录等内容没有明显丢失。

### 3. 构建本地知识库
```
python -m src.build.run
```
主要处理过程：
1. src/chunker/chunk_loader.py：读取章节 TXT，生成结构化 Chunk。

2. src/chunker/secondary_chunker.py：根据文本长度和语义边界进行二次切分，保留必要的标题上下文和文本重叠。

3. 使用 src/models/bge_m3_embedder.py 加载本地 BGE-M3。

4. 通过 src/embedding/embedding_pipeline.py 批量生成向量。

5. 将向量写入 FAISS 索引。

6. 保存与向量对应的 Chunk 元数据。

检查构建结果，确认 FAISS 索引与 Chunk 元数据文件均已生成。

例如，若当前配置使用 data/faiss/，应检查：
```aiignore
data/faiss/
├── index.faiss
└── chunks.json
```


实际目录和文件名以 vector_store.py 的保存配置为准。

需要确认 FAISS 索引中的向量数量与元数据中的 Chunk 数量一致，且向量维度与检索时使用的模型一致。


### 4. 运行检索评测
使用项目已有的评测入口运行批量测试。当前已知的候选入口为：
```aiignore
python -m src.evaluator.run_evaluation
```


执行前请检查该脚本的参数解析和配置，确认评测集路径、索引目录和输出目录。

评测集可以包含：

- query：用户问题。

- expected_documents：预期命中的文档标识。

- expected_chunk_ids：预期命中的 Chunk ID。

- category：问题类别。

- difficulty：问题难度。

重点分析：

- Recall@K：正确文档或目标 Chunk 是否被召回。

- Precision@K：返回结果中相关结果所占比例。

- MRR：首个相关结果的排名表现。

对失败样例进行分类，判断问题来自 PDF 解析、分块、向量检索还是排序。

### 5. 构建 BM25 检索

当前版本已实现 BM25 模块，先确保依赖已安装：
```aiignore
pip install rank-bm25 jieba
```


BM25 应使用与 Dense Retrieval 相同的 Retrieval Chunks 构建索引，并对语料和查询采用一致的文本处理方式。

建议逐步完善：

- jieba 中文分词。

- 汽车领域词典。

- 标准编号归一化。

- BM25 Top-K 查询测试。

- 使用同一份评测集比较 BM25 与 Dense Retrieval。

### 6. 更新知识库（可选）

新增或修改标准 PDF 后，按照依赖关系重新执行：

1. PDF 解析与章节拆分。

2. 构建本地知识库

3. 运行检索评测

4. 构建 BM25 检索


不要将新旧版本的向量与 Chunk 元数据混合使用。