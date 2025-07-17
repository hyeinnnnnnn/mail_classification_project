import sys
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer, util
import pymysql
import re
import torch
import pandas as pd  # pandas 라이브러리 임포트

# openpyxl이 설치되어 있지 않다면, 터미널에서 다음 명령어로 설치하세요:
# pip install openpyxl

# 모델 및 형태소 분석기 초기화
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()

# DB 연결 (키워드 테이블 삽입/업데이트를 위해 필요)
try:
    conn = pymysql.connect(
        host='10.50.131.18',
        port=3306,
        user='user',
        password='user1234',
        database='maildb',  # 키워드 테이블이 있는 데이터베이스
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


# 현재 테이블을 제외한 다른 테이블에서 키워드 존재 여부 확인
def keyword_exists_in_any_other_table(current_table_name, word):
    for table_name in keyword_tables:
        if table_name == current_table_name:  # 현재 테이블은 건너뜀
            continue
        cursor.execute(f"SELECT 1 FROM {table_name} WHERE word = %s", (word,))
        if cursor.fetchone() is not None:
            return True  # 다른 테이블에서 발견됨
    return False  # 다른 테이블에서는 발견되지 않음


if __name__ == "__main__":
    # --- XLSX 파일에서 메일 제목 읽어오기 ---
    # 여기에 XLSX 파일 경로를 지정하세요.
    # 예: "C:/Users/YourUser/Documents/government_mail_samples_1000.xlsx"
    xlsx_file_path = "goverment_title_detail.xlsx"
    titles_to_process = []
    try:
        # pd.read_excel()을 사용하여 XLSX 파일 읽기
        df_mails = pd.read_excel(xlsx_file_path)
        if 'title' in df_mails.columns:
            # 제목 컬럼의 데이터를 리스트로 변환
            titles_to_process = df_mails['title'].tolist()
            print(f"✔️ '{xlsx_file_path}'에서 {len(titles_to_process)}개의 메일 제목을 성공적으로 읽어왔습니다.")
        else:
            print(f"⛔ 오류: '{xlsx_file_path}' 파일에 'title' 컬럼이 없습니다. 컬럼 이름을 확인해주세요.")
            conn.close()
            sys.exit(1)
    except FileNotFoundError:
        print(f"⛔ 오류: '{xlsx_file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        print(f"   혹시 'openpyxl' 라이브러리가 설치되지 않았다면, 'pip install openpyxl'을 실행해보세요.")
        conn.close()
        sys.exit(1)
    except Exception as e:
        print(f"⛔ XLSX 파일 읽기 중 오류 발생: {e}")
        print(f"   혹시 'openpyxl' 라이브러리가 설치되지 않았다면, 'pip install openpyxl'을 실행해보세요.")
        conn.close()
        sys.exit(1)
    # ------------------------------------

    if not titles_to_process:
        print("처리할 메일 제목이 없습니다.")
        conn.close()
        sys.exit(0)

    MAX_KEYWORDS_PER_TABLE = 300
    MIN_SIMILARITY_FOR_KEYWORD_ADDITION = 0.45

    for mail_idx, title in enumerate(titles_to_process):
        print(f"\n--- {mail_idx + 1}/{len(titles_to_process)} Processing Mail Title for Keyword Learning ---")
        # title이 NaN(Not a Number)일 경우 문자열로 변환하여 처리
        display_title = str(title) if len(str(title)) < 70 else str(title)[:67] + "..."
        print(f"  Mail Title: '{display_title}'")

        # pandas가 읽은 데이터 중 NaN 값이 있을 수 있으므로 문자열 타입인지 확인
        if not isinstance(title, str):
            print(f"⛔ Skipping non-string or NaN title: {title}")
            continue

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

                # Check if max_sims_from_title_token_to_best_cat is empty
                if max_sims_from_title_token_to_best_cat.numel() > 0:
                    idx_of_best_candidate_token = torch.argmax(max_sims_from_title_token_to_best_cat).item()
                    keyword_to_add = candidate_tokens[idx_of_best_candidate_token]
                else:
                    print(
                        f"  ⚠️  No maximum similarity found for candidate tokens in '{target_table_name}'. Skipping keyword selection.")
                    continue

                print(f"  ✨ Selected keyword from title to add/update: '{keyword_to_add}'")

                existing_sim_in_target_table = get_keyword_and_similarity_from_table(target_table_name, keyword_to_add)
                print(
                    f"  DEBUG: Existing similarity for '{keyword_to_add}' in '{target_table_name}': {existing_sim_in_target_table}")

                if existing_sim_in_target_table is not None:
                    if highest_avg_sim_for_category > existing_sim_in_target_table:
                        print(
                            f"  DEBUG: New average similarity ({highest_avg_sim_for_category:.4f}) > Existing similarity ({existing_sim_in_target_table:.4f}). Attempting update in {target_table_name}.")
                        if not keyword_exists_in_any_other_table(target_table_name, keyword_to_add):
                            update_keyword_similarity(target_table_name, keyword_to_add, highest_avg_sim_for_category)
                        else:
                            print(
                                f"  ⛔ Skipped update for '{keyword_to_add}' in '{target_table_name}'. It already exists in another table.")
                    else:
                        print(
                            f"  🔁 Token '{keyword_to_add}' already exists in '{target_table_name}' with higher or equal average similarity ({existing_sim_in_target_table:.4f}). Skipping update.")
                else:
                    print(
                        f"  DEBUG: Keyword '{keyword_to_add}' does not exist in '{target_table_name}'. Attempting insertion.")
                    if not keyword_exists_in_any_other_table(target_table_name, keyword_to_add):
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