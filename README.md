# ChatRAG

基于本地PDF文件构建智能问答系统

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


### 注意
1. 后续使用若需进一步提升效果，可以根据文档格式修改文档解析方法。

### todo
1. 中英文支持
2. OCR识别
