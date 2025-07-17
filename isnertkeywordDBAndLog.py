"""
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
        if not keyword_rows:
            print(f"⚠️  Table {table} is empty. Skipping.")
            continue

        max_similarity = 0.0
        best_token = None

        for i, token in enumerate(tokens):
            token_embedding = token_embeddings[i]
            for _, word, _ in keyword_rows:
                word_embedding = model.encode(word, convert_to_tensor=True)
                similarity = float(util.pytorch_cos_sim(token_embedding, word_embedding))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_token = token

        if best_token:
            if max_similarity <= 0.5:
                print(f"🧊 Token '{best_token}' skipped for {table} (similarity {max_similarity:.2f} too low).")
            elif keyword_exists(table, best_token):
                print(f"🔁 Token '{best_token}' already exists in {table}. Skipping.")
            else:
                if count_table_entries(table) >= 300:
                    delete_lowest_similarity_keyword(table)
                    print(f"🗑 Deleted lowest similarity keyword from {table} to make room.")
                insert_keyword(table, best_token, max_similarity)
                print(f"✅ Inserted '{best_token}' into {table} (similarity: {max_similarity:.2f})")

# 연결 종료
cursor.close()
conn.close()
"""


#메일 타이틀에서 토큰을 추출하여 키워드와 유사도 계산하는 방식
'''
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql
import re
import torch  # torch.mean을 사용하기 위해 임포트

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
    return [token.form for token in tokens if token.tag in ("NNG", "NNP") and token.form.isalpha()]


def get_keywords_from_table(table_name):
    cursor.execute(f"SELECT keywordnum, word, similarity FROM {table_name}")
    return cursor.fetchall()


# 키워드가 존재하면 해당 키워드의 현재 유사도를 반환하고, 없으면 None을 반환
# 이제 이 함수는 키워드의 존재 여부만 확인하고, 실제 유사도 값은 사용하지 않을 수 있습니다.
# 하지만 일관성을 위해 유지하거나, 필요에 따라 word만 반환하는 함수로 변경 가능합니다.
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


def keyword_exists_in_any_other_table(current_table_name, word):
    for table_name in keyword_tables:
        if table_name == current_table_name: # 현재 처리 중인 테이블은 건너김
            continue
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE word = %s", (word,))
        if cursor.fetchone() is not None:
            return True # 다른 테이블에서 발견됨
    return False # 다른 테이블에서는 발견되지 않음

# 메일 제목 전부 가져오기
cursor.execute("SELECT detail FROM mail_info")
titles = [row[0] for row in cursor.fetchall()]

# 전체 처리
for title in titles:
    print(f"\n📨 Processing title: {title}")
    tokens = extract_keywords(title)
    if not tokens:
        print("⛔ No valid tokens found.")
        continue

    # 메일 제목의 모든 토큰 임베딩
    title_token_embeddings = model.encode(tokens, convert_to_tensor=True)

    best_table_for_title = None
    max_avg_similarity_for_title = -1.0  # 메일 제목과 테이블 간의 최고 평균 유사도
    best_token_for_table_insertion = None  # 실제로 삽입/업데이트될 키워드

    # 각 키워드 테이블별 평균 유사도 계산
    for table_name in keyword_tables:
        keyword_rows = get_keywords_from_table(table_name)
        if not keyword_rows:
            print(f"⚠️  Table {table_name} is empty. Skipping average similarity calculation for this table.")
            continue

        table_words = [row[1] for row in keyword_rows]
        table_word_embeddings = model.encode(table_words, convert_to_tensor=True)

        # 메일 제목의 모든 토큰과 해당 테이블의 모든 키워드 간의 유사도 매트릭스 계산
        # shape: (len(title_tokens), len(table_keywords))
        similarity_matrix = util.pytorch_cos_sim(title_token_embeddings, table_word_embeddings)

        # 각 메일 제목 토큰별로 테이블 내 키워드들과의 최대 유사도를 찾음 (가장 관련성 높은 키워드만 고려)
        # shape: (len(title_tokens),)
        max_similarities_per_title_token = similarity_matrix.max(dim=1)[0]

        # 해당 테이블의 모든 키워드를 고려한 메일 제목의 평균 유사도
        if len(max_similarities_per_title_token) > 0:
            avg_similarity_for_table = torch.mean(max_similarities_per_title_token).item()
        else:
            avg_similarity_for_table = 0.0  # 유사도 계산할 토큰이 없으면 0으로 간주

        print(f"📊 Average similarity for '{table_name}': {avg_similarity_for_table:.2f}")

        # 현재 테이블이 지금까지 찾은 최고 평균 유사도를 가진 테이블이라면 업데이트
        if avg_similarity_for_table > max_avg_similarity_for_title:
            max_avg_similarity_for_title = avg_similarity_for_table
            best_table_for_title = table_name

            # 이 테이블에 삽입될 가장 적합한 단일 키워드를 찾습니다.
            # 이는 메일 제목의 토큰 중, 해당 테이블의 키워드들과 가장 높은 유사도를 보이는 토큰입니다.
            if len(max_similarities_per_title_token) > 0:
                idx_of_best_token_in_title = torch.argmax(max_similarities_per_title_token).item()
                best_token_for_table_insertion = tokens[idx_of_best_token_in_title]
            else:
                best_token_for_table_insertion = None

    # 메일 제목 처리 완료 후, 가장 높은 평균 유사도를 보인 테이블에 단일 키워드 반영
    # 또한, 최소한의 평균 유사도 임계치를 넘어야만 반영하도록 합니다.
    MIN_AVG_SIMILARITY_THRESHOLD = 0.35  # 이 값을 조절하여 민감도를 조정할 수 있습니다.

    # 전체 처리
    for title in titles:
        # ... (기존 메일 제목 처리 로직 생략) ...

        if best_table_for_title and best_token_for_table_insertion and max_avg_similarity_for_title >= MIN_AVG_SIMILARITY_THRESHOLD:
            target_table = best_table_for_title
            best_token = best_token_for_table_insertion
            similarity_to_save = max_avg_similarity_for_title

            print(f"🎯 Selected table: {target_table} (Avg Sim: {max_avg_similarity_for_title:.2f})")
            print(f"✨ Candidate keyword for insertion/update: '{best_token}'")

            # 1. 먼저, 현재 선택된 테이블에 키워드가 존재하는지 확인
            if keyword_exists(target_table, best_token):
                existing_similarity = get_existing_keyword_similarity(target_table, best_token)

                if similarity_to_save > existing_similarity:  # 새 평균 유사도가 기존 유사도보다 높다면 업데이트
                    # 2. 다른 테이블에 동일한 키워드가 없는지 확인 후 업데이트
                    if not keyword_exists_in_any_other_table(target_table, best_token):
                        update_keyword_similarity(target_table, best_token, similarity_to_save)
                        print(
                            f"🔄 Updated '{best_token}' in {target_table} with new average similarity: {similarity_to_save:.2f} (was: {existing_similarity:.2f})")
                    else:
                        print(
                            f"⛔ Skipped update for '{best_token}' in {target_table}. It already exists in another table.")
                else:
                    print(
                        f"🔁 Token '{best_token}' already exists in {target_table} with higher or equal average similarity ({existing_similarity:.2f}). Skipping update.")
            else:  # 키워드가 현재 테이블에 존재하지 않는다면 새로 삽입
                # 2. 다른 테이블에 동일한 키워드가 없는지 확인 후 삽입
                if not keyword_exists_in_any_other_table(target_table, best_token):
                    if count_table_entries(target_table) >= 300:
                        delete_lowest_similarity_keyword(target_table)
                        print(f"🗑 Deleted lowest similarity keyword from {target_table} to make room.")
                    insert_keyword(target_table, best_token, similarity_to_save)
                    print(
                        f"✅ Inserted '{best_token}' into {target_table} (average similarity: {similarity_to_save:.2f})")
                else:
                    print(
                        f"⛔ Skipped insertion for '{best_token}' in {target_table}. It already exists in another table.")
        else:
            print(
                f"🤷 No sufficiently relevant table found for this title (Max Avg Sim: {max_avg_similarity_for_title:.2f}, Threshold: {MIN_AVG_SIMILARITY_THRESHOLD}).")
# 연결 종료
cursor.close()
conn.close()
'''

import sys
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql
import re
import torch

# 모델 및 형태소 분석기 초기화
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()

# DB 연결
try:
    conn = pymysql.connect(
        host='10.50.131.18',
        port=3306,
        user='user',
        password='user1234',
        database='maildb',
        charset='utf8'
    )
    cursor = conn.cursor()

except pymysql.Error as e:
    print(f"DB 연결 오류: {e}")
    sys.exit(1)

# 키워드 테이블 목록 (카테고리 목록)
keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']


# 명사/형용사만 추출 (NNG: 일반 명사, NNP: 고유 명사, VA: 형용사)
def extract_nouns_and_adjectives(text):
    tokens = kiwi.tokenize(text)
    return [token.form for token in tokens if token.tag in ("NNG", "NNP", "VA") and token.form.isalpha()]


def get_keywords_from_table(table_name):
    cursor.execute(f"SELECT word FROM {table_name}")
    return [row[0] for row in cursor.fetchall()]


def get_keyword_and_similarity_from_table(table_name, word):
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
        print(f"🗑 Deleted lowest similarity keyword from {table_name} to make room.")
        return True
    return False


def insert_keyword(table_name, word, similarity):
    try:
        cursor.execute(
            f"INSERT INTO {table_name}(word, similarity) VALUES (%s, %s)",
            (word, similarity)
        )
        conn.commit()
        print(f"✅ Inserted '{word}' into {table_name} (similarity: {similarity:.4f})")
        return True
    except pymysql.Error as e:
        print(f"DB 삽입 오류 ({table_name}, {word}): {e}")
        conn.rollback()
        return False


def update_keyword_similarity(table_name, word, new_similarity):
    try:
        cursor.execute(
            f"UPDATE {table_name} SET similarity = %s WHERE word = %s",
            (new_similarity, word)
        )
        conn.commit()
        print(f"🔄 Updated '{word}' in {table_name} with new similarity: {new_similarity:.4f}")
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류 ({table_name}, {word}): {e}")
        conn.rollback()
        return False


# 수정된 함수: 현재 테이블을 제외한 다른 테이블에서 키워드 존재 여부 확인
def keyword_exists_in_any_other_table(current_table_name, word):
    """
    현재 처리 중인 테이블을 제외한 나머지 키워드 테이블에서 특정 키워드의 존재 여부를 확인합니다.
    """
    for table_name in keyword_tables:
        if table_name == current_table_name:  # 현재 테이블은 건너뜀
            continue
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE word = %s", (word,))
        if cursor.fetchone() is not None:
            return True  # 다른 테이블에서 발견됨
    return False  # 다른 테이블에서는 발견되지 않음


if __name__ == "__main__":
    try:
        cursor.execute("SELECT detail FROM mail_info")
        titles_to_process = [row[0] for row in cursor.fetchall()]
    except pymysql.Error as e:
        print(f"메일 제목 조회 오류: {e}")
        conn.close()
        sys.exit(1)

    if not titles_to_process:
        print("처리할 메일 제목이 없습니다.")
        conn.close()
        sys.exit(0)

    MAX_KEYWORDS_PER_TABLE = 300
    MIN_SIMILARITY_FOR_KEYWORD_ADDITION = 0.5

    for mail_idx, title in enumerate(titles_to_process):
        print(f"\n--- {mail_idx + 1}/{len(titles_to_process)} Processing Mail Title for Keyword Learning ---")
        display_title = title if len(title) < 70 else title[:67] + "..."
        print(f"  Mail Title: '{display_title}'")

        candidate_tokens = extract_nouns_and_adjectives(title)
        print(f"  DEBUG: Extracted candidate tokens: {candidate_tokens}")
        if not candidate_tokens:
            print(
                "⛔ No valid noun/adjective tokens found in this mail title. Skipping keyword learning for this title.")
            continue

        candidate_token_embeddings = model.encode(candidate_tokens, convert_to_tensor=True)

        current_all_table_keywords = {}
        current_all_keyword_embeddings = {}

        for table_name in keyword_tables:
            keywords_in_table = get_keywords_from_table(table_name)
            current_all_table_keywords[table_name] = keywords_in_table
            if keywords_in_table:
                current_all_keyword_embeddings[table_name] = model.encode(keywords_in_table, convert_to_tensor=True)
            else:
                current_all_keyword_embeddings[table_name] = torch.tensor([])

        best_category_for_title_semantic = None
        highest_avg_sim_for_category = -1.0

        for table_name in keyword_tables:
            embeddings_in_table = current_all_keyword_embeddings[table_name]

            if not embeddings_in_table.numel():
                print(f"  ⚠️  Category '{table_name}' has no embeddings. Skipping similarity calculation.")
                continue

            similarity_matrix = util.pytorch_cos_sim(candidate_token_embeddings, embeddings_in_table)
            max_sim_per_candidate_token_to_category = similarity_matrix.max(dim=1)[0]

            if len(max_sim_per_candidate_token_to_category) > 0:
                avg_sim_for_current_category = torch.mean(max_sim_per_candidate_token_to_category).item()
            else:
                avg_sim_for_current_category = 0.0

            print(f"  Category '{table_name}': Average Max Sim = {avg_sim_for_current_category:.4f}")

            if avg_sim_for_current_category > highest_avg_sim_for_category:
                highest_avg_sim_for_category = avg_sim_for_current_category
                best_category_for_title_semantic = table_name

        print(
            f"  DEBUG: Final best category candidate: '{best_category_for_title_semantic}' with Avg Sim: {highest_avg_sim_for_category:.4f}")
        print(f"  DEBUG: Threshold for addition: {MIN_SIMILARITY_FOR_KEYWORD_ADDITION:.4f}")

        if best_category_for_title_semantic and highest_avg_sim_for_category >= MIN_SIMILARITY_FOR_KEYWORD_ADDITION:
            target_table_name = best_category_for_title_semantic
            print(
                f"  🎯 Best category for this title: '{target_table_name}' (Overall Avg Max Sim: {highest_avg_sim_for_category:.4f})")

            table_embeddings_for_best_category = current_all_keyword_embeddings[target_table_name]

            keyword_to_add = None
            if table_embeddings_for_best_category.numel() and len(candidate_tokens) > 0:
                final_similarity_matrix_for_best_cat = util.pytorch_cos_sim(candidate_token_embeddings,
                                                                            table_embeddings_for_best_category)
                max_sims_from_title_token_to_best_cat = final_similarity_matrix_for_best_cat.max(dim=1)[0]

                idx_of_best_candidate_token = torch.argmax(max_sims_from_title_token_to_best_cat).item()
                keyword_to_add = candidate_tokens[idx_of_best_candidate_token]

                print(f"  ✨ Selected keyword from title to add/update: '{keyword_to_add}'")

                existing_sim_in_target_table = get_keyword_and_similarity_from_table(target_table_name, keyword_to_add)
                print(
                    f"  DEBUG: Existing similarity for '{keyword_to_add}' in '{target_table_name}': {existing_sim_in_target_table}")

                if existing_sim_in_target_table is not None:
                    # 현재 테이블에 키워드가 이미 존재하고, 새 평균 유사도가 더 높다면
                    if highest_avg_sim_for_category > existing_sim_in_target_table:
                        # 다른 테이블에 동일한 키워드가 없는지 확인 후 업데이트
                        if not keyword_exists_in_any_other_table(target_table_name, keyword_to_add):
                            print(
                                f"  DEBUG: New average similarity ({highest_avg_sim_for_category:.4f}) > Existing similarity ({existing_sim_in_target_table:.4f}). Attempting update in {target_table_name}.")
                            update_keyword_similarity(target_table_name, keyword_to_add, highest_avg_sim_for_category)
                        else:
                            print(
                                f"  ⛔ Skipped update for '{keyword_to_add}' in '{target_table_name}'. It already exists in another table.")
                    else:
                        print(
                            f"  🔁 Token '{keyword_to_add}' already exists in '{target_table_name}' with higher or equal average similarity ({existing_sim_in_target_table:.4f}). Skipping update.")
                else:
                    # 키워드가 현재 테이블에 존재하지 않는 경우
                    # 다른 테이블에 동일한 키워드가 없는지 확인 후 삽입
                    if not keyword_exists_in_any_other_table(target_table_name, keyword_to_add):
                        print(
                            f"  DEBUG: Keyword '{keyword_to_add}' does not exist in '{target_table_name}' and not in other tables. Attempting insertion.")
                        if count_table_entries(target_table_name) >= MAX_KEYWORDS_PER_TABLE:
                            delete_lowest_similarity_keyword(target_table_name)
                        insert_keyword(target_table_name, keyword_to_add, highest_avg_sim_for_category)
                    else:
                        print(
                            f"  ⛔ Skipped insertion for '{keyword_to_add}' in '{target_table_name}'. It already exists in another table.")
            else:
                print(
                    f"  ⚠️  No keywords in '{target_table_name}' or no candidate tokens from title to compare with. Cannot determine best candidate keyword for insertion.")
        else:
            print(
                f"  🤷 No sufficiently relevant category found for this title (Max Avg Sim: {highest_avg_sim_for_category:.4f}, Threshold: {MIN_SIMILARITY_FOR_KEYWORD_ADDITION:.4f}). Skipping keyword learning.")

cursor.close()
conn.close()
print(f"\n--- 키워드 학습 프로세스 완료 ---")