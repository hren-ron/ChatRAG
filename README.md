# ChatRAG

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