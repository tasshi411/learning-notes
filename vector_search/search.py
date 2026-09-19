"""
Qdrantに格納したベクトルを検索する

事前準備: docker compose up -d && uv run ingest.py
実行:     uv run search.py
"""
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# ingest.py と同じ条件でベクトル化しないと正しく検索できない
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "company_regulations"
GPT_5_6_MODEL_NAME="gpt-5.6-luna"

load_dotenv()
qdrant_client = QdrantClient(url=QDRANT_URL)
openai_client = OpenAI()  # 環境変数 OPENAI_API_KEY を自動で読みこむ


def get_text_embedding(text: str) -> list[float]:
    """
    textの中身をベクトルに変換して返す
    """
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIM)
    return response.data[0].embedding


def generate_pseudo_answer(query: str) -> str:
    """
    LLMで擬似的な回答を生成する
    """
    # 実際は検索結果や生成AIなどを用いて疑似回答を作る処理を書く
    prompt=f'''
下記の問い合わせに関して、擬似的な回答を1~2文で回答してください。
問い合わせ: "{query}"
'''
    return openai_client.responses.create(model=GPT_5_6_MODEL_NAME, input=prompt).output_text

def generate_answer(query: str, chunks: list[str]):
    """
    ベクトル検索で取得したドキュメントをもとに検索を行う
    """
    references = ""
    for i, chunk in enumerate(chunks):
        references += f"{i+1}: `{chunk}"
    prompt = f'''
下記の問い合わせに関して、参考資料を元に回答を生成してください。
**問い合わせ**
{query}

**参考資料**
{references}
'''
    return openai_client.responses.create(model=GPT_5_6_MODEL_NAME, input=prompt).output_text

# 1. クエリとなる文章を宣言
query_text = "年休はどんくらいつく？"
query_text = generate_pseudo_answer(query_text)
print(query_text)

# 2. クエリ文章をベクトル化
query_vector = get_text_embedding(query_text)

# 3. Qdrantで類似度の高い順に検索
results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3,
).points

# 4. 結果表示
print(f'"{query_text}"の検索結果')
for point in results:
    print(f"\n[{point.score:.3f}] {point.payload['text']}")

# 5. テキスト生成
