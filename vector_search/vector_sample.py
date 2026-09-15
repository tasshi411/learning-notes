import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()
client = OpenAI()  # 環境変数 OPENAI_API_KEY を自動で読みこむ

def get_text_embedding(text: str):
    """
    textの中身をベクトルに変換して返す。
    """
    return client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    2つのベクトルのコサイン類似度を返す
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# 1. 検索対象の文章を宣言
# texts = [
#     "猫が好きです",
#     "犬を飼っています",
#     "空が青いです",
#     "お腹がすきました"
# ]
texts = [
    "猫が好きです",
    "犬を飼っています",
    "犬を買っています",
    "空が青いです",
    "お腹がすきました"
]

# 2. 検索対象の文章をベクトル化
vectors = [get_text_embedding(text) for text in texts]

# 3. ベクトルの情報を表示
print(len(vectors[0].data[0].embedding)) # ベクトルの次元数を表示
# print(vectors[0]) # ベクトルを表示

# 4. クエリとなる文章を宣言
# query_text = "空の色は？"
query_text = "動物の値段は？"

# 5. クエリ文章をベクトル化
query_vector = get_text_embedding(query_text)

# 6. 類似度を計算(コサイン類似度）
results = [cosine_similarity(query_vector.data[0].embedding, vector.data[0].embedding) for vector in vectors]

# 7. 結果表示
print(f'"{query_text}"と各検索対象の文章との類似度')
for text, result in zip(texts, results):
    print(f'{text}: {result}')
