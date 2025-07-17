from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql

# 모델 로드 (Ko-sBERT: 의미 기반 한국어 문장 임베딩)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()
# DB 연결
conn = pymysql.connect(
    host='10.50.131.18',
    port = 3306,
    user='user',
    password='user1234',
    database='maildb',
    charset='utf8'
)
cursor = conn.cursor()

# 키워드 테이블 목록
keyword_tables = ["finance_keywords", "government_keywords", "portal_keywords", "advertisement_keywords"]

# 각 키워드 테이블의 키워드 불러오기 함수
def fetch_keywords(table):
    cursor.execute(f"SELECT keywordnum, word FROM {table}")
    return cursor.fetchall()

# mail_info 테이블에서 전체 제목 가져오기
cursor.execute("SELECT title FROM mail_info")
titles = [row[0] for row in cursor.fetchall()]

# 제목 하나씩 처리
for title in titles:
    tokens = [token.form for token in kiwi.tokenize(title) if len(token.form) > 1]
    token_embeddings = model.encode(tokens, convert_to_tensor=True)

    for i, token in enumerate(tokens):
        token_vector = token_embeddings[i]

        for table in keyword_tables:
            keywords = fetch_keywords(table)
            if not keywords:
                continue

            keyword_words = [kw[1] for kw in keywords]
            keyword_vectors = model.encode(keyword_words, convert_to_tensor=True)

            similarities = util.cos_sim(token_vector, keyword_vectors).squeeze()
            max_sim_idx = similarities.argmax().item()
            max_sim = similarities[max_sim_idx].item()

            if max_sim >= 0.5:
                matched_keyword = keyword_words[max_sim_idx]

                # 중복 확인
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE word = %s", (token,))
                exists = cursor.fetchone()[0]

                if exists:
                    # 유사도 갱신 (기존보다 높을 경우에만)
                    cursor.execute(
                        f"UPDATE {table} SET similarity = GREATEST(similarity, %s) WHERE word = %s",
                        (max_sim, token)
                    )
                else:
                    # 새로운 키워드 삽입
                    cursor.execute(
                        f"INSERT INTO {table} (word, similarity) VALUES (%s, %s)",
                        (token, max_sim)
                    )
                conn.commit()
                break  # 한 테이블에만 삽입하고 종료
    break
cursor.close()
conn.close()