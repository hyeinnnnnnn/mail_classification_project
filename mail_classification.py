
import sys
from sentence_transformers import SentenceTransformer, util
import pymysql
import torch
from collections import defaultdict
import re
from kiwipiepy import Kiwi

# 모델 초기화 (KR-SBERT 모델)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

# DB 연결
try:
    conn = pymysql.connect(
        host='10.50.131.18',
        port=3306,
        user='user',
        password='user1234',
        database='maildb',
        charset='utf8mb4'
    )
    cursor = conn.cursor()

except pymysql.Error as e:
    print(f"DB 연결 오류: {e}")
    sys.exit(1)

# --- 상수 정의 ---
MAX_TOP_KEYWORDS_GLOBAL = 50
MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION = 0.2
FINAL_CLASSIFICATION_THRESHOLD = 0.5


# -----------------

def get_category_id_by_name(category_name):
    """
    Categories 테이블에서 category_name에 해당하는 category_id를 조회
    """
    try:
        cursor.execute("SELECT category_id FROM Categories WHERE category_name = %s", (category_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None
    except pymysql.Error as e:
        print(f"category_id 조회 오류 (Category: {category_name}): {e}")
        return None


def update_mail_category(user_id, mail_num, classified_value):
    """
    mail_info 테이블의 'categori' 컬럼에 분류된 값을 업데이트
    classified_value는 category_id (INT) 또는 'Unclassified' (STR) 등
    """
    try:
        cursor.execute(
            f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
            (classified_value, user_id, mail_num)
        )
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Value: '{classified_value}'): {e}")
        conn.rollback()
        return False


def extract_meaningful_tokens(text):
    """
    텍스트에서 명사, 고유 명사를 추출
    """
    if not text:
        return []

    tokens = kiwi.tokenize(text)
    meaningful_words = []
    # NNG: 일반 명사, NNP: 고유 명사
    allowed_pos = ["NNG", "NNP"]
    for token in tokens:
        if token.tag in allowed_pos and len(token.form) > 1:
            meaningful_words.append(token.form)
    return meaningful_words


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python mail_classification.py <target_user_id>")
        print("예시: python mail_classification.py user123")
        sys.exit(1)

    target_user_id = sys.argv[1]
    print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")

    # --- official_mail_domains 정보 로드 (2차 분류용) ---
    official_domains_map = {}
    print("✔️ official_mail_domains 테이블에서 공식 메일 주소/도메인 정보 로드 중...")
    try:
        cursor.execute("""
            SELECT omd.domain, c.category_name
            FROM official_mail_domains omd
            JOIN Categories c ON omd.category_id = c.category_id
        """)
        for domain_or_email, category_name in cursor.fetchall():
            official_domains_map[domain_or_email.lower()] = category_name
        print(f"✔️ 총 {len(official_domains_map)}개의 공식 메일 주소/도메인 정보 로드 완료.")
    except pymysql.Error as e:
        print(f"공식 메일 주소/도메인 정보 로드 오류: {e}")
        conn.close()
        sys.exit(1)

    # --- SpecialKeywords 데이터 로드 (1차 분류용) ---
    special_keywords_map = {}
    print("✔️ SpecialKeywords 테이블에서 특별 키워드 정보 로드 중...")
    try:
        cursor.execute("""
            SELECT sk.keyword_text, c.category_name
            FROM SpecialKeywords sk
            JOIN Categories c ON sk.category_id = c.category_id
        """)
        for keyword, category_name in cursor.fetchall():
            special_keywords_map[keyword.lower()] = category_name  # 모두 소문자로 저장
        print(f"✔️ 총 {len(special_keywords_map)}개의 특별 키워드 정보 로드 완료.")
    except pymysql.Error as e:
        print(f"SpecialKeywords 데이터 로드 오류: {e}")
        conn.close()
        sys.exit(1)

    # --- 모든 카테고리-소분류-키워드 데이터를 한 번에 로드 및 임베딩 (3차 분류용) ---
    global_keywords_texts = []
    global_keyword_meta_data = []
    keyword_text_to_global_idx = {}

    print("Keywords 데이터를 로드 중...")
    try:
        cursor.execute("""
            SELECT
                k.keyword_text,
                c.category_name
            FROM Keywords k
            JOIN Subcategories s ON k.subcategory_id = s.subcategory_id
            JOIN Categories c ON s.category_id = c.category_id
        """)
        db_keywords_data = cursor.fetchall()

        if not db_keywords_data:
            print("DB에 분류할 키워드 데이터가 없습니다. 스크립트를 종료합니다.")
            conn.close()
            sys.exit(1)

        for keyword_text, category_name in db_keywords_data:
            if keyword_text not in keyword_text_to_global_idx:
                keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
                global_keywords_texts.append(keyword_text)
            global_keyword_meta_data.append((keyword_text, category_name, keyword_text_to_global_idx[keyword_text]))

        print(f"총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
        global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
        print("모든 키워드 임베딩 생성 완료.")

    except pymysql.Error as e:
        print(f"키워드 데이터 로드 오류: {e}")
        conn.close()
        sys.exit(1)
    # ----------------------------------------------------------------------------------

    try:
        cursor.execute("SELECT user_id, mailnum, title, detail, categori, sentBy FROM mail_info WHERE user_id = %s",
                       (target_user_id,))
        mails_to_process = cursor.fetchall()
    except pymysql.Error as e:
        print(f"메일 조회 오류: {e}")
        conn.close()
        sys.exit(1)

    if not mails_to_process:
        print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
        conn.close()
        sys.exit(0)

    for mail_idx, (user_id, mailnum, title, detail, current_categori, sentBy_email) in enumerate(mails_to_process):
        print(
            f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
        display_title = title if len(title) < 70 else title[:67] + "..."
        print(f"  Mail Title: '{display_title}'")

        # --- 0. 이미 분류된 메일은 건너뛰기 ---
        if current_categori is not None and str(current_categori).strip().lower() != "unclassified":
            print(f"  Mail (MailNum: {mailnum}) already classified as '{current_categori}'. Skipping.")
            continue
        # ------------------------------------

        classified_by_any_tier = False  # 어떤 분류 단계에서든 성공했는지 플래그

        # --- 1. (0차 분류 실패 시) 메일 제목에 DB에 등록된 특별 키워드 포함 시 1차 분류 시도 ---
        mail_title_lower = title.lower() if title else ""

        for special_keyword_text, category_name_from_map in special_keywords_map.items():
            if special_keyword_text in mail_title_lower:
                classified_category_name = category_name_from_map
                category_id_special_keyword = get_category_id_by_name(classified_category_name)

                if category_id_special_keyword is not None:
                    if update_mail_category(user_id, mailnum, category_id_special_keyword):
                        print(
                            f"  1차 분류 (특별 키워드 - 제목): Detected '{special_keyword_text}' in mail title. Classifying as '{classified_category_name}' (ID: {category_id_special_keyword}).")
                        classified_by_any_tier = True
                        break
                    else:
                        print(f"  1차 분류 실패 (특별 키워드 - 제목): DB 업데이트 실패. 다음 분류 시도.")
                else:
                    print(
                        f"  경고: 특별 키워드 '{special_keyword_text}'에 대한 카테고리 ID '{classified_category_name}'를 찾을 수 없습니다. 다음 분류 시도.")

        if classified_by_any_tier:
            continue

        # --- 2. (1차 분류 실패 시) 발신자 이메일 주소 (sentBy)에 공식 도메인/메일 주소가 포함되는지 2차 분류 시도 ---
        if sentBy_email:
            sentBy_email_lower = sentBy_email.lower()
            for registered_domain_or_email, category_name_from_map in official_domains_map.items():
                if registered_domain_or_email in sentBy_email_lower:
                    classified_category_name = category_name_from_map
                    category_id_for_inclusion = get_category_id_by_name(classified_category_name)

                    if category_id_for_inclusion is not None:
                        if update_mail_category(user_id, mailnum, category_id_for_inclusion):
                            print(
                                f"  2차 분류 (공식 메일 주소/도메인): Mail classified as '{classified_category_name}' (ID: {category_id_for_inclusion}) because '{registered_domain_or_email}' is in '{sentBy_email}'.")
                            classified_by_any_tier = True
                            break
                        else:
                            print(f"  2차 분류 실패 (공식 메일 주소/도메인): DB 업데이트 실패. 다음 분류 시도.")
                    else:
                        print(f"  2차 분류 (공식 메일 주소/도메인): 카테고리 '{classified_category_name}'에 대한 ID를 찾을 수 없습니다. 다음 분류 시도.")

            if classified_by_any_tier:
                continue
            else:
                print(f"  2차 분류 (공식 메일 주소/도메인): '{sentBy_email}'에 포함되는 공식 도메인/메일 주소를 찾지 못했습니다. 다음 분류 시도.")
        else:
            print(f"  2차 분류 (공식 메일 주소/도메인): 발신자 이메일 주소 (sentBy) 없음. 다음 분류 시도.")

        # --- 3. (1, 2차 분류 모두 실패 시) 키워드 기반 분류 실행 (기존 로직) ---
        combined_mail_text = (title if title else "") + " " + (detail if detail else "")
        extracted_keywords_for_mail = extract_meaningful_tokens(combined_mail_text)

        if not extracted_keywords_for_mail:
            classified_category_name = "Unclassified (No_Keywords_Extracted_from_Combined_Text)"
            print(f"  3차 분류 (키워드): No meaningful keywords extracted. Classifying as: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
            continue

        print(f"  3차 분류 (키워드): Extracted keywords for mail: {extracted_keywords_for_mail}")

        extracted_keyword_embeddings = model.encode(extracted_keywords_for_mail, convert_to_tensor=True)

        all_similarities_matrix = util.pytorch_cos_sim(extracted_keyword_embeddings, global_keyword_embeddings)

        keyword_similarity_list = []
        for i, extracted_kw_text_from_mail in enumerate(extracted_keywords_for_mail):
            for keyword_text_from_db, category_name, global_idx in global_keyword_meta_data:
                similarity = all_similarities_matrix[i, global_idx].item()
                if similarity > MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION:
                    keyword_similarity_list.append(
                        (similarity, extracted_kw_text_from_mail, keyword_text_from_db, category_name))

        keyword_similarity_list.sort(key=lambda x: x[0], reverse=True)

        top_n_similar_pairs = keyword_similarity_list[:MAX_TOP_KEYWORDS_GLOBAL]

        category_scores_sum = defaultdict(float)
        category_scores_count = defaultdict(int)

        for sim, _, _, category_name in top_n_similar_pairs:
            category_scores_sum[category_name] += sim
            category_scores_count[category_name] += 1

        final_category_averages = {}
        for category, total_sum in category_scores_sum.items():
            final_category_averages[category] = total_sum / category_scores_count[category]

        print(
            f"  DEBUG: Top {MAX_TOP_KEYWORDS_GLOBAL} (유사도 > {MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION}) 키워드 쌍 기반 카테고리별 평균 유사도: {final_category_averages}")

        best_category_name = None
        highest_avg_similarity = -1.0

        if not final_category_averages:
            classified_category_name = "Unclassified (No_Strong_Keyword_Matches_Above_Threshold)"
            print(f"  3차 분류 (키워드): No meaningful keywords extracted. Classifying as: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
            continue

        for category_name, avg_sim in final_category_averages.items():
            if avg_sim > highest_avg_similarity:
                highest_avg_similarity = avg_sim
                best_category_name = category_name

        if best_category_name and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
            classified_category_name = best_category_name
            category_id_final = get_category_id_by_name(classified_category_name)
            if category_id_final is not None:
                if update_mail_category(user_id, mailnum, category_id_final):
                    print(
                        f"  3차 분류 (키워드): Classification Result: {classified_category_name} (ID: {category_id_final}, Highest Avg Sim: {highest_avg_similarity:.2f})")
                else:
                    print(f"  3차 분류 실패 (키워드): DB 업데이트 실패. 'Unclassified'로 설정 시도.")
                    if update_mail_category(user_id, mailnum, "Unclassified"):
                        print(f"UserID {user_id}, MailNum {mailnum} updated to 'Unclassified'.")
            else:
                print(f"  경고: 최종 분류된 카테고리 '{classified_category_name}'에 대한 ID를 찾을 수 없습니다. 'Unclassified'로 설정.")
                if update_mail_category(user_id, mailnum, "Unclassified"):
                    print(f"UserID {user_id}, MailNum {mailnum} updated to 'Unclassified'.")
        else:
            classified_category_name = "Unclassified (Below_Final_Classification_Threshold)"
            print(f"  3차 분류 (키워드): Classification Result: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to 'Unclassified'.")

cursor.close()
conn.close()
print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")