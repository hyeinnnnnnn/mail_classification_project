from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql
import re

# 모델 및 형태소 분석기 초기화
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()

# DB 연결
conn = pymysql.connect(
    host='10.50.131.18',
    port=3306,
    user='user',
    password='user1234',
    database='maildb',
    charset='utf8'
)
cursor = conn.cursor()

# 키워드 테이블 목록
keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']

# 명사/형용사만 추출
def extract_keywords(text):
    tokens = kiwi.tokenize(text)
    return [token.form for token in tokens if token.tag in ("NNG", "NNP", "VA") and token.form.isalpha()]

def get_keywords_from_table(table_name):
    cursor.execute(f"SELECT keywordnum, word, similarity FROM {table_name}")
    return cursor.fetchall()

# 키워드가 존재하면 해당 키워드의 현재 유사도를 반환하고, 없으면 None을 반환
def get_existing_keyword_similarity(table_name, word):
    cursor.execute(f"SELECT similarity FROM {table_name} WHERE word = %s", (word,))
    result = cursor.fetchone()
    return result[0] if result else None

def count_table_entries(table_name):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

def delete_lowest_similarity_keyword(table_name):
    cursor.execute(f"SELECT keywordnum FROM {table_name} ORDER BY similarity ASC LIMIT 1")
    result = cursor.fetchone()
    if result:
        cursor.execute(f"DELETE FROM {table_name} WHERE keywordnum = %s", (result[0],))
        conn.commit()

def insert_keyword(table_name, word, similarity):
    cursor.execute(
        f"INSERT INTO {table_name}(word, similarity) VALUES (%s, %s)",
        (word, similarity)
    )
    conn.commit()

def update_keyword_similarity(table_name, word, new_similarity):
    cursor.execute(
        f"UPDATE {table_name} SET similarity = %s WHERE word = %s",
        (new_similarity, word)
    )
    conn.commit()


# 메일 제목 전부 가져오기
cursor.execute("SELECT title FROM mail_info")
titles = [row[0] for row in cursor.fetchall()]

# 전체 처리
for title in titles:
    print(f"\n📨 Processing title: {title}")
    tokens = extract_keywords(title)
    if not tokens:
        print("⛔ No valid tokens found.")
        continue

    token_embeddings = model.encode(tokens, convert_to_tensor=True)

    for table in keyword_tables:
        keyword_rows = get_keywords_from_table(table)
        # 테이블이 비어있는 경우를 대비하여 비어있지 않도록 처리
        if not keyword_rows:
            print(f"⚠️  Table {table} is empty. No keywords to compare against. Skipping this table for now.")
            continue  # 다음 테이블로 넘어감

        max_similarity = 0.0
        best_token = None

        # 현재 테이블의 키워드 임베딩을 미리 계산
        table_words = [row[1] for row in keyword_rows]
        if not table_words:  # 테이블에 단어가 없는 경우 다시 확인
            continue  # 다음 테이블로 넘어감

        table_word_embeddings = model.encode(table_words, convert_to_tensor=True)

        for i, token in enumerate(tokens):
            token_embedding = token_embeddings[i]
            # 추출된 토큰과 테이블 내 모든 키워드 간의 유사도 계산
            similarities = util.pytorch_cos_sim(token_embedding, table_word_embeddings)

            # 현재 토큰에 대한 테이블 내 최대 유사도 찾기
            current_max_similarity = similarities.max().item()  # .item()으로 텐서 값을 Python 숫자로 변환

            if current_max_similarity > max_similarity:
                max_similarity = current_max_similarity
                best_token = token

        if best_token:
            existing_similarity = get_existing_keyword_similarity(table, best_token)

            if max_similarity <= 0.5:
                print(f"🧊 Token '{best_token}' skipped for {table} (similarity {max_similarity:.2f} too low).")
            elif existing_similarity is not None:  # 키워드가 이미 존재한다면
                if max_similarity > existing_similarity:
                    update_keyword_similarity(table, best_token, max_similarity)
                    print(
                        f"🔄 Updated '{best_token}' in {table} with higher similarity: {max_similarity:.2f} (was: {existing_similarity:.2f})")
                else:
                    print(
                        f"🔁 Token '{best_token}' already exists in {table} with higher or equal similarity ({existing_similarity:.2f}). Skipping update.")
            else:  # 키워드가 존재하지 않는다면 새로 삽입
                if count_table_entries(table) >= 300:
                    delete_lowest_similarity_keyword(table)
                    print(f"🗑 Deleted lowest similarity keyword from {table} to make room.")
                insert_keyword(table, best_token, max_similarity)
                print(f"✅ Inserted '{best_token}' into {table} (similarity: {max_similarity:.2f})")

# 연결 종료
cursor.close()
conn.close()