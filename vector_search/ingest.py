"""
社内規定のMarkdownを見出し3（条・FAQ）単位でベクトル化し、Qdrantに格納する。

事前準備: docker compose up -d
実行:     uv run ingest.py
"""
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 512
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "company_regulations"
SOURCE_FILE = Path(__file__).parent / "docs" / "社内規定_クレストシステムズ.md"

load_dotenv()
openai_client = OpenAI()  # 環境変数 OPENAI_API_KEY を自動で読みこむ


def parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    """
    先頭の --- で囲まれたfrontmatterを辞書にして、本文と分けて返す
    """
    match = re.match(r"^---\n(.*?)\n---\n", markdown, re.DOTALL)
    if not match:
        return {}, markdown
    meta = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, markdown[match.end():]


def split_by_h3(body: str) -> list[dict]:
    """
    見出し3単位でチャンクに分割する。
    「## 第N章 ○○」配下の見出し3のみを対象とし、章番号と章タイトルを付与する。
    """
    chunks = []
    chapter_number, chapter_title = None, None
    heading, lines = None, []

    def flush():
        if heading and chapter_number is not None:
            # 区切り線（---）と前後の空行を除去
            content = "\n".join(l for l in lines if l.strip() != "---").strip()
            print(content)
            print("\n")
            chunks.append({
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                # 章名・見出しもベクトルに反映させるため本文の先頭に付ける
                "text": f"第{chapter_number}章 {chapter_title} > {heading}\n\n{content}",
            })

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            heading, lines = None, []
            m = re.match(r"## 第(\d+)章\s+(.+)", line)
            chapter_number, chapter_title = (int(m.group(1)), m.group(2).strip()) if m else (None, None)
        elif line.startswith("### "):
            flush()
            heading, lines = line[4:].strip(), []
        elif heading:
            lines.append(line)
    flush()
    return chunks


def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    """
    複数のtextをまとめてベクトルに変換して返す
    """
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts, dimensions=EMBEDDING_DIM)
    return [d.embedding for d in response.data]


def main():
    meta, body = parse_frontmatter(SOURCE_FILE.read_text(encoding="utf-8"))
    chunks = split_by_h3(body)
    print(f"{len(chunks)}件のチャンクを作成しました")

    vectors = get_text_embeddings([c["text"] for c in chunks])

    qdrant = QdrantClient(url=QDRANT_URL)
    # 再実行時は作り直す
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    qdrant.create_payload_index(COLLECTION_NAME, "chapter_number", PayloadSchemaType.INTEGER)
    qdrant.create_payload_index(COLLECTION_NAME, "document_id", PayloadSchemaType.KEYWORD)

    points = [
        PointStruct(
            # 同じ文書・同じ見出しなら同じIDになるようにする
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['document_id']}/{chunk['text'].splitlines()[0]}")),
            vector=vector,
            payload={
                "chapter_number": chunk["chapter_number"],
                "chapter_title": chunk["chapter_title"],
                "text": chunk["text"],
                "document_id": meta["document_id"],
                "effective_date": meta["effective_date"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{qdrant.count(COLLECTION_NAME).count}件をコレクション「{COLLECTION_NAME}」に格納しました")


if __name__ == "__main__":
    main()
