from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql
import re
import torch

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
    # isalpha()를 추가하여 숫자나 특수문자가 포함된 토큰은 제외
    return [token.form for token in tokens if token.tag in ("NNG", "NNP", "VA") and token.form.isalpha()]


def get_keywords_from_table(table_name):
    cursor.execute(f"SELECT keywordnum, word, similarity FROM {table_name}")
    return cursor.fetchall()


def get_existing_keyword_similarity(table_name, word):
    cursor.execute(f"SELECT similarity FROM {table_name} WHERE word = %s", (word,))
    result = cursor.fetchone()
    return result[0] if result else None


def keyword_exists(table_name, word):
    cursor.execute(f"SELECT 1 FROM {table_name} WHERE word = %s", (word,))
    return cursor.fetchone() is not None


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


# 모든 키워드 테이블에서 특정 단어가 존재하는지 확인 (현재 테이블 제외)
def keyword_exists_in_any_other_table(current_table_name, word):
    for table_name in keyword_tables:
        if table_name == current_table_name:
            continue
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE word = %s", (word,))
        if cursor.fetchone() is not None:
            return True
    return False


# 메일 내용 전부 가져오기
cursor.execute("SELECT detail FROM mail_info")
details = [row[0] for row in cursor.fetchall()]

# 쿼리 카운트 초기화
query_count = 0

# --- 메인 처리 루프 ---
for detail in details:
    query_count += 1
    # 메일 내용이 너무 길면 잘라서 출력
    display_detail = detail if len(detail) < 100 else detail[:97] + "..."
    print(f"\n--- Processing Mail {query_count} ---")
    print(f"  Content: '{display_detail}'")

    tokens = extract_keywords(detail)
    if not tokens:
        print("  ⛔ No valid tokens (nouns/adjectives) found for this content.")
        continue

    detail_token_embeddings = model.encode(tokens, convert_to_tensor=True)

    best_table_for_detail = None
    max_avg_similarity_for_detail = -1.0  # 메일 내용과 테이블 간의 최고 평균 유사도
    best_token_for_table_insertion = None  # 실제로 삽입/업데이트될 키워드 (해당 테이블과의 개별 유사도 기준)

    # 각 키워드 테이블별 평균 유사도 계산 및 최적 테이블 선택
    for table_name in keyword_tables:
        keyword_rows = get_keywords_from_table(table_name)
        if not keyword_rows:
            # print(f"⚠️  Table {table_name} is empty. Skipping average similarity calculation for this table.")
            continue

        table_words = [row[1] for row in keyword_rows]
        table_word_embeddings = model.encode(table_words, convert_to_tensor=True)

        similarity_matrix = util.pytorch_cos_sim(detail_token_embeddings, table_word_embeddings)
        max_similarities_per_detail_token = similarity_matrix.max(dim=1)[0]

        if len(max_similarities_per_detail_token) > 0:
            avg_similarity_for_table = torch.mean(max_similarities_per_detail_token).item()
        else:
            avg_similarity_for_table = 0.0

        # 현재 테이블이 지금까지 찾은 최고 평균 유사도를 가진 테이블이라면 업데이트
        if avg_similarity_for_table > max_avg_similarity_for_detail:
            max_avg_similarity_for_detail = avg_similarity_for_table
            best_table_for_detail = table_name

            # 이 테이블에 삽입될 가장 적합한 단일 키워드를 찾습니다.
            # 이는 메일 내용의 토큰 중, 해당 테이블의 키워드들과 가장 높은 유사도를 보이는 토큰입니다.
            if len(max_similarities_per_detail_token) > 0:
                idx_of_best_token_in_detail = torch.argmax(max_similarities_per_detail_token).item()
                best_token_for_table_insertion = tokens[idx_of_best_token_in_detail]
            else:
                best_token_for_table_insertion = None

    # 최소 평균 유사도 임계치 (이 값을 조절하여 민감도를 조정할 수 있습니다.)
    MIN_AVG_SIMILARITY_THRESHOLD = 0.35

    # 최종 선택된 테이블에 키워드 반영 (삽입 또는 업데이트)
    if best_table_for_detail and best_token_for_table_insertion and max_avg_similarity_for_detail >= MIN_AVG_SIMILARITY_THRESHOLD:
        target_table = best_table_for_detail
        best_token = best_token_for_table_insertion
        similarity_to_save = max_avg_similarity_for_detail  # DB에 저장될 유사도 값은 테이블의 평균 유사도

        # 키워드가 이미 타겟 테이블에 존재하는지 확인
        if keyword_exists(target_table, best_token):
            existing_similarity = get_existing_keyword_similarity(target_table, best_token)

            if similarity_to_save > existing_similarity:  # 새 평균 유사도가 기존 유사도보다 높다면 업데이트
                # 다른 테이블에 동일한 키워드가 없는지 확인 후 업데이트 (중복 방지)
                if not keyword_exists_in_any_other_table(target_table, best_token):
                    update_keyword_similarity(target_table, best_token, similarity_to_save)
                    print(
                        f"  ✅ Updated keyword: '{best_token}' in {target_table} (New Avg Sim: {similarity_to_save:.2f}, Prev: {existing_similarity:.2f})")
                else:
                    print(
                        f"  ⛔ Skipped update for '{best_token}' in {target_table}. It already exists in another table.")
            else:
                print(
                    f"  ↔️ Keyword: '{best_token}' in {target_table} (Avg Sim: {existing_similarity:.2f}). No higher average similarity found.")
        else:  # 키워드가 현재 테이블에 존재하지 않는다면 새로 삽입
            # 다른 테이블에 동일한 키워드가 없는지 확인 후 삽입 (중복 방지)
            if not keyword_exists_in_any_other_table(target_table, best_token):
                if count_table_entries(target_table) >= 300:  # 테이블 용량 초과 시 가장 낮은 유사도 키워드 삭제
                    delete_lowest_similarity_keyword(target_table)
                    print(f"  🗑 Deleted lowest similarity keyword from {target_table} to make room.")
                insert_keyword(target_table, best_token, similarity_to_save)
                print(f"  ➕ Inserted keyword: '{best_token}' into {target_table} (Avg Sim: {similarity_to_save:.2f})")
            else:
                print(
                    f"  ⛔ Skipped insertion for '{best_token}' in {target_table}. It already exists in another table.")
    else:
        # 이 줄에서 변수명을 max_avg_similarity_for_detail로 수정했습니다.
        print(
            f"  🤷 No sufficiently relevant table found for this content (Max Avg Sim: {max_avg_similarity_for_detail:.2f}, Threshold: {MIN_AVG_SIMILARITY_THRESHOLD}).")

# --- DB 연결 종료 ---
cursor.close()
conn.close()