import jieba


class AutomotiveTokenizer:

    def __init__(self, dictionary_path: str):

        self.tokenizer = jieba.Tokenizer()
        self.tokenizer.load_userdict(dictionary_path)

    def tokenize(self, text: str):
        return [
            word.strip()
            for word in self.tokenizer.cut(text)
            if word.strip()
        ]