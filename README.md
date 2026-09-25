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
在.env中配置模型信息（API_KEY, BASE_URL）

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


### 注意
1. 后续使用若需进一步提升效果，可以根据文档格式修改文档解析方法。

### todo
1. 中英文支持
2. OCR识别
