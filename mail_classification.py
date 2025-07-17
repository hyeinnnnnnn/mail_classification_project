# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# def get_keywords_from_table(table_name):
#     # 키워드 가져오는 함수
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
# def update_mail_category(user_id, mail_num, category_result):
#     # mail_info 테이블의 categori 컬럼 업데이트
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
# if __name__ == "__main__":
#     # 명령줄 인자로부터 target_user_id (분류할 사용자 ID) 받기
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]  # 스크립트 실행 시 첫 번째 인자로 받을 user_id
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # 모든 키워드 테이블의 키워드를 미리 로드 및 임베딩 (효율성을 위해)
#     all_table_keywords = {}
#     all_keyword_embeddings = {}
#
#     for table_name in keyword_tables:
#         keywords = get_keywords_from_table(table_name)
#         all_table_keywords[table_name] = keywords
#         if keywords:
#             all_keyword_embeddings[table_name] = model.encode(keywords, convert_to_tensor=True)
#         else:
#             all_keyword_embeddings[table_name] = torch.tensor([]) # 키워드가 없을 경우 빈 텐서
#
#     # 특정 user_id의 메일 제목, user_id, mailNum을 전부 가져오기
#     try:
#         cursor.execute("SELECT user_id, mailnum, title FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()  # (user_id, mailNum, title) 튜플 리스트
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     # 분류 관련 임계치
#     # 하나의 메일 제목이 여러 카테고리의 키워드와 유사할 수 있으므로,
#     # 카테고리별 최고 유사도를 기준으로 최종 분류를 결정합니다.
#     FINAL_CLASSIFICATION_THRESHOLD = 0.6  # 최종 분류로 인정하기 위한 최소 유사도 (이 미만이면 Unclassified)
#
#     # --- 메인 분류 및 DB 업데이트 루프 ---
#     for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
#         print(f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         # 메일 제목 전체를 임베딩
#         mail_title_embedding = model.encode(title, convert_to_tensor=True)
#
#         category_max_similarities = {} # 각 카테고리의 최고 유사도 저장
#
#         # 메일 제목 임베딩과 각 카테고리 키워드 임베딩 간의 유사도 계산
#         for table_name in keyword_tables:
#             embeddings_in_table = all_keyword_embeddings[table_name]
#
#             if not embeddings_in_table.numel(): # 해당 카테고리에 키워드가 없는 경우
#                 continue
#
#             # 메일 제목 임베딩 (1개)과 해당 카테고리 키워드 임베딩들 간의 코사인 유사도 계산
#             # 결과는 [1, num_keywords_in_table] 형태의 텐서
#             similarity_scores = util.pytorch_cos_sim(mail_title_embedding, embeddings_in_table)
#
#             # 해당 카테고리의 키워드들 중 메일 제목과 가장 높은 유사도를 가진 값
#             max_sim_for_category = torch.max(similarity_scores).item()
#             category_max_similarities[table_name] = max_sim_for_category
#
#         # 가장 높은 유사도를 가진 카테고리 찾기
#         if not category_max_similarities:
#             classified_category = "Unclassified (No_Keywords_Loaded)"
#             print(f"No keywords loaded for any category. Classifying as: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         best_category = None
#         highest_similarity = -1.0 # 초기값으로 가장 낮은 유사도 설정
#
#         for category, sim in category_max_similarities.items():
#             if sim > highest_similarity:
#                 highest_similarity = sim
#                 best_category = category
#
#         # 최종 분류 임계치 확인
#         if best_category and highest_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             print(f"  Classification Result: {classified_category} (Highest Similarity: {highest_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             # 유사도가 낮아서 분류되지 않은 경우 구체적인 이유 추가
#             if highest_similarity < FINAL_CLASSIFICATION_THRESHOLD:
#                 classified_category += f" (Low_Max_Sim:{highest_similarity:.2f})"
#             print(f"  Classification Result: {classified_category}")
#
#         # DB 업데이트
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# # --- DB 연결 종료 ---
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")

"""
import sys
from sentence_transformers import SentenceTransformer, util
import pymysql
import torch
from collections import defaultdict
import re # 정규표현식 모듈 추가

# 모델 초기화 (KR-SBERT 모델)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

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

def get_keywords_from_table(table_name):
    # 주어진 테이블에서 키워드를 가져오는 함수
    cursor.execute(f"SELECT word FROM {table_name}")
    return [row[0] for row in cursor.fetchall()]

def update_mail_category(user_id, mail_num, category_result):
    # mail_info 테이블의 categori 컬럼 업데이트
    try:
        cursor.execute(
            f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
            (category_result, user_id, mail_num)
        )
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
        conn.rollback()
        return False

if __name__ == "__main__":
    # 명령줄 인자로부터 target_user_id (분류할 사용자 ID) 받기
    if len(sys.argv) < 2:
        print("사용법: python mail_classification.py <target_user_id>")
        print("예시: python mail_classification.py user123")
        sys.exit(1)

    target_user_id = sys.argv[1]  # 스크립트 실행 시 첫 번째 인자로 받을 user_id
    print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")

    # 모든 키워드 테이블의 키워드를 미리 로드 및 임베딩 (효율성을 위해)
    all_table_keywords = {}
    all_keyword_embeddings = {}

    for table_name in keyword_tables:
        keywords = get_keywords_from_table(table_name)
        all_table_keywords[table_name] = keywords
        if keywords:
            all_keyword_embeddings[table_name] = model.encode(keywords, convert_to_tensor=True)
        else:
            all_keyword_embeddings[table_name] = torch.tensor([]) # 키워드가 없을 경우 빈 텐서

    # 특정 user_id의 메일 제목, user_id, mailNum을 전부 가져오기
    try:
        cursor.execute("SELECT user_id, mailnum, title FROM mail_info WHERE user_id = %s", (target_user_id,))
        mails_to_process = cursor.fetchall()  # (user_id, mailNum, title) 튜플 리스트
    except pymysql.Error as e:
        print(f"메일 조회 오류: {e}")
        conn.close()
        sys.exit(1)

    if not mails_to_process:
        print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
        conn.close()
        sys.exit(0)

    # 분류 관련 임계치
    FINAL_CLASSIFICATION_THRESHOLD = 0.4  # 최종 분류로 인정하기 위한 최소 유사도 (이 미만이면 Unclassified)

    # --- 메인 분류 및 DB 업데이트 루프 ---
    for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
        print(f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
        display_title = title if len(title) < 70 else title[:67] + "..."
        print(f"  Mail Title: '{display_title}'")

        # (광고) 포함 시 무조건 광고로 분류
        # re.search를 사용하여 대소문자 구분 없이 (광고)를 찾도록 함
        if re.search(r'\(광고\)', title, re.IGNORECASE):
            classified_category = "advertisement_keywords"
            print(f"  !!! Detected '(광고)' in title. Classifying as: {classified_category} !!!")
            if update_mail_category(user_id, mailnum, classified_category):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
            continue # 다음 메일로 넘어감

        # 메일 제목 전체를 임베딩
        mail_title_embedding = model.encode(title, convert_to_tensor=True)

        # 각 카테고리의 최고 유사도 저장
        category_max_similarities = {}

        # 메일 제목 임베딩과 각 카테고리 키워드 임베딩 간의 유사도 계산
        for table_name in keyword_tables:


            embeddings_in_table = all_keyword_embeddings[table_name]

            if not embeddings_in_table.numel(): # 해당 카테고리에 키워드가 없는 경우
                continue

            # 메일 제목 임베딩 (1개)과 해당 카테고리 키워드 임베딩들 간의 코사인 유사도 계산
            # 결과는 [1, num_keywords_in_table] 형태의 텐서
            similarity_scores = util.pytorch_cos_sim(mail_title_embedding, embeddings_in_table)

            # 해당 카테고리의 키워드들 중 메일 제목과 가장 높은 유사도를 가진 값
            max_sim_for_category = torch.max(similarity_scores).item()
            category_max_similarities[table_name] = max_sim_for_category

        # 가장 높은 유사도를 가진 카테고리 찾기
        if not category_max_similarities:
            classified_category = "Unclassified (No_Keywords_Loaded)"
            print(f"No keywords loaded for any category. Classifying as: {classified_category}")
            if update_mail_category(user_id, mailnum, classified_category):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
            continue

        best_category = None
        highest_similarity = -1.0 # 초기값으로 가장 낮은 유사도 설정

        for category, sim in category_max_similarities.items():
            if sim > highest_similarity:
                highest_similarity = sim
                best_category = category

        # 최종 분류 임계치 확인
        if best_category and highest_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
            classified_category = best_category
            print(f"  Classification Result: {classified_category} (Highest Similarity: {highest_similarity:.2f})")
        else:
            classified_category = "Unclassified"
            print(f"  Classification Result: {classified_category}")

        # DB 업데이트
        if update_mail_category(user_id, mailnum, classified_category):
            print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
        else:
            print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")

# --- DB 연결 종료 ---
cursor.close()
conn.close()
print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
"""


# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
# import re # 정규표현식 모듈 추가
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# def get_keywords_from_table(table_name):
#     # 주어진 테이블에서 키워드를 가져오는 함수
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
# def update_mail_category(user_id, mail_num, category_result):
#     # mail_info 테이블의 categori 컬럼 업데이트
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
# if __name__ == "__main__":
#     # 명령줄 인자로부터 target_user_id (분류할 사용자 ID) 받기
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]  # 스크립트 실행 시 첫 번째 인자로 받을 user_id
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # 모든 키워드 테이블의 키워드를 미리 로드 및 임베딩 (효율성을 위해)
#     all_table_keywords = {}
#     all_keyword_embeddings = {}
#
#     for table_name in keyword_tables:
#         keywords = get_keywords_from_table(table_name)
#         all_table_keywords[table_name] = keywords
#         if keywords:
#             all_keyword_embeddings[table_name] = model.encode(keywords, convert_to_tensor=True)
#         else:
#             all_keyword_embeddings[table_name] = torch.tensor([]) # 키워드가 없을 경우 빈 텐서
#
#     # 특정 user_id의 메일 제목, user_id, mailNum을 전부 가져오기
#     try:
#         cursor.execute("SELECT user_id, mailnum, title FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()  # (user_id, mailNum, title) 튜플 리스트
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     # 분류 관련 임계치
#     # 주의: 평균 유사도를 사용하므로, 이 임계치 값은 재조정해야 할 수 있습니다.
#     FINAL_CLASSIFICATION_THRESHOLD = 0.3  # 평균 유사도에 맞는 새로운 임계치 (예시 값)
#
#     # --- 메인 분류 및 DB 업데이트 루프 ---
#     for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
#         print(f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         # (광고) 포함 시 무조건 광고로 분류
#         if re.search(r'\(광고\)', title, re.IGNORECASE):
#             classified_category = "advertisement_keywords"
#             print(f"  !!! Detected '(광고)' in title. Classifying as: {classified_category} !!!")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue # 다음 메일로 넘어감
#
#         # 메일 제목 전체를 임베딩
#         mail_title_embedding = model.encode(title, convert_to_tensor=True)
#
#         # 각 카테고리의 평균 유사도 저장 (변수명 변경)
#         category_avg_similarities = {}
#
#         # 메일 제목 임베딩과 각 카테고리 키워드 임베딩 간의 유사도 계산
#         for table_name in keyword_tables:
#             embeddings_in_table = all_keyword_embeddings[table_name]
#
#             if not embeddings_in_table.numel(): # 해당 카테고리에 키워드가 없는 경우
#                 continue
#
#             # 메일 제목 임베딩 (1개)과 해당 카테고리 키워드 임베딩들 간의 코사인 유사도 계산
#             similarity_scores = util.pytorch_cos_sim(mail_title_embedding, embeddings_in_table)
#
#             # --- 변경된 부분: 최고 유사도 대신 평균 유사도 계산 ---
#             # 해당 카테고리의 키워드들 중 메일 제목과 모든 유사도의 '평균'을 가진 값
#             avg_sim_for_category = torch.mean(similarity_scores).item()
#             category_avg_similarities[table_name] = avg_sim_for_category
#             # ----------------------------------------------------
#
#         # 가장 높은 '평균 유사도'를 가진 카테고리 찾기 (변수명 변경)
#         if not category_avg_similarities:
#             classified_category = "Unclassified (No_Keywords_Loaded)"
#             print(f"No keywords loaded for any category. Classifying as: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         best_category = None
#         highest_avg_similarity = -1.0 # 초기값으로 가장 낮은 유사도 설정
#
#         for category, sim in category_avg_similarities.items(): # 변경된 딕셔너리 사용
#             if sim > highest_avg_similarity:
#                 highest_avg_similarity = sim
#                 best_category = category
#
#         # 최종 분류 임계치 확인
#         if best_category and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             # 출력 메시지 변경
#             print(f"  Classification Result: {classified_category} (Highest Average Similarity: {highest_avg_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             # 미분류 이유를 더 명확하게 표시
#             if highest_avg_similarity < FINAL_CLASSIFICATION_THRESHOLD:
#                 classified_category += f" (Low_Avg_Sim:{highest_avg_similarity:.2f})"
#             print(f"  Classification Result: {classified_category}")
#
#         # DB 업데이트
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# # --- DB 연결 종료 ---
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
''''''
# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
# import re
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# # --- 새로운 상수 정의 ---
# MAX_TOP_KEYWORDS_GLOBAL = 200  # 전체 유사도 중 상위 N개의 키워드를 선택할 개수
#
#
# # -------------------------
#
# def get_keywords_from_table(table_name):
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
#
# def update_mail_category(user_id, mail_num, category_result):
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # --- 모든 카테고리의 키워드를 한 번에 로드 및 임베딩 (전역 키워드 리스트 생성) ---
#     # 각 키워드가 어느 카테고리에 속하는지 정보도 함께 저장합니다.
#     global_keywords_texts = []  # 임베딩을 생성할 유니크한 키워드 텍스트 리스트
#     # (텍스트, 카테고리 이름, global_keywords_texts 내 해당 텍스트의 인덱스) 튜플 리스트
#     global_keyword_meta_data = []
#
#     # 중복 키워드 임베딩 방지를 위한 맵
#     keyword_text_to_global_idx = {}
#
#     print("✔️ 모든 카테고리 키워드를 로드 중...")
#     for table_name in keyword_tables:
#         keywords_from_table = get_keywords_from_table(table_name)
#         for keyword_text in keywords_from_table:
#             # 해당 키워드 텍스트가 global_keywords_texts에 이미 있는지 확인
#             if keyword_text not in keyword_text_to_global_idx:
#                 keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
#                 global_keywords_texts.append(keyword_text)
#
#             # 메타데이터에는 중복된 키워드 텍스트라도 해당 카테고리 정보를 함께 저장
#             global_keyword_meta_data.append((keyword_text, table_name, keyword_text_to_global_idx[keyword_text]))
#
#     print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
#     if not global_keywords_texts:
#         print("⛔ 분류할 키워드가 DB에 없습니다. 스크립트를 종료합니다.")
#         conn.close()
#         sys.exit(1)
#
#     global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
#     print("✔️ 모든 키워드 임베딩 생성 완료.")
#     # ----------------------------------------------------------------------------------
#
#     try:
#         cursor.execute("SELECT user_id, mailnum, detail FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     # 최종 분류 임계치 (평균 유사도를 기준으로 하므로 이 값은 조정이 필요할 수 있습니다.)
#     FINAL_CLASSIFICATION_THRESHOLD = 0.3
#
#     # --- 메인 분류 및 DB 업데이트 루프 ---
#     for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
#         print(
#             f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         # (광고) 포함 시 무조건 광고로 분류 (우선 순위가 가장 높음)
#         if re.search(r'\(광고\)', title, re.IGNORECASE):
#             classified_category = "advertisement_keywords"
#             print(f"  !!! Detected '(광고)' in title. Classifying as: {classified_category} !!!")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         # 메일 제목 임베딩
#         mail_title_embedding = model.encode(title, convert_to_tensor=True)
#
#         # --- 변경된 핵심 로직: 전체 키워드와 유사도 계산 후 Top N 선택 ---
#         # 메일 제목 임베딩과 모든 전역 키워드 임베딩 간의 코사인 유사도 계산
#         # 결과는 [1, 전체_키워드_개수] 형태의 텐서
#         all_similarities = util.pytorch_cos_sim(mail_title_embedding, global_keyword_embeddings).flatten()
#
#         # 각 키워드와 그 유사도 점수, 그리고 원래 속했던 카테고리를 묶음
#         # (유사도, 키워드 텍스트, 카테고리 이름) 튜플 리스트 생성
#         keyword_similarity_list = []
#         for keyword_text, category_name, global_idx in global_keyword_meta_data:
#             similarity = all_similarities[global_idx].item()
#             keyword_similarity_list.append((similarity, keyword_text, category_name))
#
#         # 유사도 기준으로 내림차순 정렬
#         keyword_similarity_list.sort(key=lambda x: x[0], reverse=True)
#
#         # 상위 N개 키워드 선택
#         top_n_keywords_with_sim = keyword_similarity_list[:MAX_TOP_KEYWORDS_GLOBAL]
#
#         # 선택된 Top N 키워드들을 카테고리별로 그룹화하고 유사도 합계/개수 저장
#         category_scores_sum = defaultdict(float)
#         category_scores_count = defaultdict(int)
#
#         for sim, _, category_name in top_n_keywords_with_sim:
#             category_scores_sum[category_name] += sim
#             category_scores_count[category_name] += 1
#
#         # 카테고리별 평균 유사도 계산
#         final_category_averages = {}
#         for category, total_sum in category_scores_sum.items():
#             final_category_averages[category] = total_sum / category_scores_count[category]
#
#         print(f"  DEBUG: Top {MAX_TOP_KEYWORDS_GLOBAL} 키워드 기반 카테고리별 평균 유사도: {final_category_averages}")
#         # -------------------------------------------------------------------
#
#         # 가장 높은 '평균 유사도'를 가진 카테고리 찾기
#         best_category = None
#         highest_avg_similarity = -1.0
#
#         if not final_category_averages:  # 상위 N개 키워드에 포함된 카테고리가 없을 경우
#             classified_category = "Unclassified (No_Top_Keywords_Matched)"
#             print(f"  Classification Result: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         for category, avg_sim in final_category_averages.items():
#             if avg_sim > highest_avg_similarity:
#                 highest_avg_similarity = avg_sim
#                 best_category = category
#
#         # 최종 분류 임계치 확인
#         if best_category and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             print(
#                 f"  Classification Result: {classified_category} (Highest Average Sim from Top {MAX_TOP_KEYWORDS_GLOBAL} : {highest_avg_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             print(f"  Classification Result: {classified_category}")
#
#         # DB 업데이트
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# # --- DB 연결 종료 ---
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
''''''
# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
# import re
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# # --- 새로운 상수 정의 ---
# MAX_TOP_KEYWORDS_GLOBAL = 200  # 전체 유사도 중 상위 N개의 키워드를 선택할 개수
# MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION = 0.2  # 이 유사도 이하인 키워드는 아예 고려하지 않음
#
#
# # -------------------------
#
# def get_keywords_from_table(table_name):
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
#
# def update_mail_category(user_id, mail_num, category_result):
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # --- 모든 카테고리의 키워드를 한 번에 로드 및 임베딩 (전역 키워드 리스트 생성) ---
#     global_keywords_texts = []
#     global_keyword_meta_data = []
#     keyword_text_to_global_idx = {}
#
#     print("✔️ 모든 카테고리 키워드를 로드 중...")
#     for table_name in keyword_tables:
#         keywords_from_table = get_keywords_from_table(table_name)
#         for keyword_text in keywords_from_table:
#             if keyword_text not in keyword_text_to_global_idx:
#                 keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
#                 global_keywords_texts.append(keyword_text)
#
#             global_keyword_meta_data.append((keyword_text, table_name, keyword_text_to_global_idx[keyword_text]))
#
#     print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
#     if not global_keywords_texts:
#         print("⛔ 분류할 키워드가 DB에 없습니다. 스크립트를 종료합니다.")
#         conn.close()
#         sys.exit(1)
#
#     global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
#     print("✔️ 모든 키워드 임베딩 생성 완료.")
#     # ----------------------------------------------------------------------------------
#
#     try:
#         cursor.execute("SELECT user_id, mailnum, title FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     # 최종 분류 임계치 (평균 유사도를 기준으로 하므로 이 값은 조정이 필요할 수 있습니다.)
#     FINAL_CLASSIFICATION_THRESHOLD = 0.4
#
#     for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
#         print(
#             f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         if re.search(r'\(광고\)', title, re.IGNORECASE):
#             classified_category = "advertisement_keywords"
#             print(f"  !!! Detected '(광고)' in title. Classifying as: {classified_category} !!!")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         mail_title_embedding = model.encode(title, convert_to_tensor=True)
#
#         all_similarities = util.pytorch_cos_sim(mail_title_embedding, global_keyword_embeddings).flatten()
#
#         keyword_similarity_list = []
#         for keyword_text, category_name, global_idx in global_keyword_meta_data:
#             similarity = all_similarities[global_idx].item()
#             # --- 변경된 부분: 유사도가 MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION(0.2)보다 커야 포함 ---
#             if similarity > MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION:
#                 keyword_similarity_list.append((similarity, keyword_text, category_name))
#             # -------------------------------------------------------------------------------------
#
#         # 유사도 기준으로 내림차순 정렬
#         keyword_similarity_list.sort(key=lambda x: x[0], reverse=True)
#
#         # 상위 N개 키워드 선택 (이제 0.2 초과 유사도만 포함됨)
#         top_n_keywords_with_sim = keyword_similarity_list[:MAX_TOP_KEYWORDS_GLOBAL]
#
#         category_scores_sum = defaultdict(float)
#         category_scores_count = defaultdict(int)
#
#         for sim, _, category_name in top_n_keywords_with_sim:
#             category_scores_sum[category_name] += sim
#             category_scores_count[category_name] += 1
#
#         final_category_averages = {}
#         for category, total_sum in category_scores_sum.items():
#             final_category_averages[category] = total_sum / category_scores_count[category]
#
#         print(
#             f"  DEBUG: Top {MAX_TOP_KEYWORDS_GLOBAL} (유사도 > {MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION}) 키워드 기반 카테고리별 평균 유사도: {final_category_averages}")
#
#         best_category = None
#         highest_avg_similarity = -1.0
#
#         if not final_category_averages:  # 상위 N개 키워드에 포함된 카테고리가 없을 경우
#             classified_category = f"Unclassified (No_Keywords_Above_{MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION}_Found)"
#             print(f"  Classification Result: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         for category, avg_sim in final_category_averages.items():
#             if avg_sim > highest_avg_similarity:
#                 highest_avg_similarity = avg_sim
#                 best_category = category
#
#         if best_category and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             print(
#                 f"  Classification Result: {classified_category} (Highest Avg Sim from Top {MAX_TOP_KEYWORDS_GLOBAL} : {highest_avg_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             # if highest_avg_similarity < FINAL_CLASSIFICATION_THRESHOLD:
#             #     classified_category += f" (Low_Avg_Sim_Top_{MAX_TOP_KEYWORDS_GLOBAL}:{highest_avg_similarity:.2f})"
#             # else:
#             #     classified_category += " (Unknown_Reason)"
#             print(f"  Classification Result: {classified_category}")
#
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
''''''
# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
# import re
# from kiwipiepy import Kiwi  # Kiwi 라이브러리 임포트
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # Kiwi 형태소 분석기 초기화
# # Kiwi는 한번 로드되면 재사용 가능하므로 전역으로 선언
# kiwi = Kiwi()
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# # --- 상수 정의 ---
# MAX_TOP_KEYWORDS_GLOBAL = 200  # 전체 유사도 중 상위 N개의 키워드를 선택할 개수
# MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION = 0.2  # 이 유사도 이하인 키워드는 아예 고려하지 않음
# FINAL_CLASSIFICATION_THRESHOLD = 0.5  # 최종 분류 임계치 (이 값은 조정이 필요할 수 있음)
#
#
# # -----------------
#
# def get_keywords_from_table(table_name):
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
#
# def update_mail_category(user_id, mail_num, category_result):
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
#
# def extract_meaningful_tokens(text):
#     """
#     Kiwi를 사용하여 텍스트에서 명사 추출
#     불필요한 품사는 제외하고 의미 있는 키워드 위주로 추출합니다.
#     """
#     tokens = kiwi.tokenize(text)
#     meaningful_words = []
#     # N: 명사, V: 동사, M: 수식언(관형사, 부사), A: 형용사 (필요에 따라 품사 태그 조정)
#     allowed_pos = ["NNG", "NNP"]
#     for token in tokens:
#         # 단어 길이가 1인 경우는 제외 (예: '이', '그', '더' 등)
#         if token.tag[0] in allowed_pos and len(token.form) > 1:
#             # 동사는 어간만 사용하는 것이 더 적합할 수 있으나,
#             # 여기서는 원형(form)을 그대로 사용하여 문맥 유지
#             meaningful_words.append(token.form)
#     return meaningful_words
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # --- 모든 카테고리의 키워드를 한 번에 로드 및 임베딩 (전역 키워드 리스트 생성) ---
#     global_keywords_texts = []
#     global_keyword_meta_data = []  # (키워드 텍스트, 카테고리 이름, global_keywords_texts 내 인덱스)
#     keyword_text_to_global_idx = {}  # 중복 방지 맵
#
#     print("✔️ 모든 카테고리 키워드를 로드 중...")
#     for table_name in keyword_tables:
#         keywords_from_table = get_keywords_from_table(table_name)
#         for keyword_text in keywords_from_table:
#             if keyword_text not in keyword_text_to_global_idx:
#                 keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
#                 global_keywords_texts.append(keyword_text)
#
#             global_keyword_meta_data.append((keyword_text, table_name, keyword_text_to_global_idx[keyword_text]))
#
#     print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
#     if not global_keywords_texts:
#         print("⛔ 분류할 키워드가 DB에 없습니다. 스크립트를 종료합니다.")
#         conn.close()
#         sys.exit(1)
#
#     global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
#     print("✔️ 모든 키워드 임베딩 생성 완료.")
#     # ----------------------------------------------------------------------------------
#
#     try:
#         cursor.execute("SELECT user_id, mailnum, title FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     for mail_idx, (user_id, mailnum, title) in enumerate(mails_to_process):
#         print(
#             f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         if re.search(r'\(광고\)', title, re.IGNORECASE):
#             classified_category = "advertisement_keywords"
#             print(f"  !!! Detected '(광고)' in title. Classifying as: {classified_category} !!!")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         # --- 변경된 부분: 메일 제목에서 키워드 추출 및 임베딩 ---
#         extracted_keywords_from_title = extract_meaningful_tokens(title)
#
#         if not extracted_keywords_from_title:
#             classified_category = "Unclassified (No_Keywords_Extracted_from_Title)"
#             print(f"  No meaningful keywords extracted from title. Classifying as: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         print(f"  Extracted keywords from title: {extracted_keywords_from_title}")
#
#         # 추출된 각 키워드를 임베딩
#         extracted_keyword_embeddings = model.encode(extracted_keywords_from_title, convert_to_tensor=True)
#         # ---------------------------------------------------------
#
#         # --- 변경된 부분: 추출된 키워드 임베딩과 전역 키워드 임베딩 간의 유사도 계산 ---
#         # 결과는 [추출된_키워드_개수, 전체_키워드_개수] 형태의 텐서
#         # 각 행은 추출된 하나의 키워드와 모든 DB 키워드 간의 유사도 점수
#         all_similarities_matrix = util.pytorch_cos_sim(extracted_keyword_embeddings, global_keyword_embeddings)
#
#         # 모든 유사도 점수를 평탄화하여 하나의 리스트로 만들고, 각 점수와 원래 키워드/카테고리 정보 결합
#         keyword_similarity_list = []
#         for i, extracted_kw_text in enumerate(extracted_keywords_from_title):
#             for keyword_text, category_name, global_idx in global_keyword_meta_data:
#                 similarity = all_similarities_matrix[i, global_idx].item()
#                 # 유사도 기준 필터링: MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION(0.2)보다 커야 포함
#                 if similarity > MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION:
#                     # 이 유사도 점수가 어떤 '추출된 키워드'와 어떤 'DB 키워드/카테고리'에서 나왔는지 기록
#                     keyword_similarity_list.append((similarity, extracted_kw_text, keyword_text, category_name))
#         # -------------------------------------------------------------------------------------
#
#         # 유사도 기준으로 내림차순 정렬 (추출된 키워드-DB키워드 쌍에 대한 유사도)
#         keyword_similarity_list.sort(key=lambda x: x[0], reverse=True)
#
#         # 상위 N개 유사도 쌍 선택
#         top_n_similar_pairs = keyword_similarity_list[:MAX_TOP_KEYWORDS_GLOBAL]
#
#         # Top N 유사도 쌍들을 카테고리별로 그룹화하고 유사도 합계/개수 저장
#         category_scores_sum = defaultdict(float)
#         category_scores_count = defaultdict(int)
#
#         # Top N에 해당하는 (DB) 키워드의 카테고리 정보를 사용
#         for sim, _, _, category_name in top_n_similar_pairs:
#             category_scores_sum[category_name] += sim
#             category_scores_count[category_name] += 1
#
#         # 카테고리별 평균 유사도 계산
#         final_category_averages = {}
#         for category, total_sum in category_scores_sum.items():
#             final_category_averages[category] = total_sum / category_scores_count[category]
#
#         print(
#             f"  DEBUG: Top {MAX_TOP_KEYWORDS_GLOBAL} (유사도 > {MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION}) 키워드 쌍 기반 카테고리별 평균 유사도: {final_category_averages}")
#
#         best_category = None
#         highest_avg_similarity = -1.0
#
#         if not final_category_averages:  # 유효한 유사도 쌍이 없거나 Top N에 들지 못함
#             classified_category = f"Unclassified (No_Strong_Keyword_Matches_Above_{MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION})"
#             print(f"  Classification Result: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         for category, avg_sim in final_category_averages.items():
#             if avg_sim > highest_avg_similarity:
#                 highest_avg_similarity = avg_sim
#                 best_category = category
#
#         if best_category and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             print(
#                 f"  Classification Result: {classified_category} (Highest Avg Sim from Top {MAX_TOP_KEYWORDS_GLOBAL} : {highest_avg_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             if highest_avg_similarity < FINAL_CLASSIFICATION_THRESHOLD:
#                 classified_category += f" (Low_Avg_Sim_Top_{MAX_TOP_KEYWORDS_GLOBAL}:{highest_avg_similarity:.2f})"
#             else:
#                 classified_category += " (Unknown_Reason)"  # 거의 발생하지 않을 경우
#             print(f"  Classification Result: {classified_category}")
#
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
''''''
# import sys
# from sentence_transformers import SentenceTransformer, util
# import pymysql
# import torch
# from collections import defaultdict
# import re
# from kiwipiepy import Kiwi  # Kiwi 라이브러리 임포트
#
# # 모델 초기화 (KR-SBERT 모델)
# model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
#
# # Kiwi 형태소 분석기 초기화
# kiwi = Kiwi()
#
# # DB 연결
# try:
#     conn = pymysql.connect(
#         host='10.50.131.18',
#         port=3306,
#         user='user',
#         password='user1234',
#         database='maildb',
#         charset='utf8'
#     )
#     cursor = conn.cursor()
#
# except pymysql.Error as e:
#     print(f"DB 연결 오류: {e}")
#     sys.exit(1)
#
# # 키워드 테이블 목록 (카테고리 목록)
# keyword_tables = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']
#
# # --- 상수 정의 ---
# MAX_TOP_KEYWORDS_GLOBAL = 100  # 전체 유사도 중 상위 N개의 키워드를 선택할 개수
# MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION = 0.2  # 이 유사도 이하인 키워드는 아예 고려하지 않음
# FINAL_CLASSIFICATION_THRESHOLD = 0.5  # 최종 분류 임계치 (이 값은 조정이 필요할 수 있음)
#
#
# # -----------------
#
# def get_keywords_from_table(table_name):
#     cursor.execute(f"SELECT word FROM {table_name}")
#     return [row[0] for row in cursor.fetchall()]
#
#
# def update_mail_category(user_id, mail_num, category_result):
#     try:
#         cursor.execute(
#             f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
#             (category_result, user_id, mail_num)
#         )
#         conn.commit()
#         return True
#     except pymysql.Error as e:
#         print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Category: {category_result}): {e}")
#         conn.rollback()
#         return False
#
#
# def extract_meaningful_tokens(text):
#     """
#     Kiwi를 사용하여 텍스트에서 명사, 동사, 형용사 등을 추출합니다.
#     불필요한 품사는 제외하고 의미 있는 키워드 위주로 추출합니다.
#     """
#     if not text:  # 입력 텍스트가 None이거나 비어있으면 빈 리스트 반환
#         return []
#
#     tokens = kiwi.tokenize(text)
#     meaningful_words = []
#     allowed_pos = ["NNG", "NNP"]
#     for token in tokens:
#         # 단어 길이가 1인 경우는 제외 (불필요한 조사나 어미일 가능성이 높으므로 제외
#         if token.tag[0] in allowed_pos and len(token.form) > 1:
#             meaningful_words.append(token.form)
#     return meaningful_words
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python mail_classification.py <target_user_id>")
#         print("예시: python mail_classification.py user123")
#         sys.exit(1)
#
#     target_user_id = sys.argv[1]
#     print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")
#
#     # --- 모든 카테고리의 키워드를 한 번에 로드 및 임베딩 (전역 키워드 리스트 생성) ---
#     # 더 빠르게 하기 위해
#     global_keywords_texts = []
#     global_keyword_meta_data = []  # (키워드 텍스트, 카테고리 이름, global_keywords_texts 내 인덱스)
#     keyword_text_to_global_idx = {}  # 중복 방지 맵
#
#     print("✔️ 모든 카테고리 키워드를 로드 중...")
#     for table_name in keyword_tables:
#         keywords_from_table = get_keywords_from_table(table_name)
#         for keyword_text in keywords_from_table:
#             if keyword_text not in keyword_text_to_global_idx:
#                 keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
#                 global_keywords_texts.append(keyword_text)
#
#             global_keyword_meta_data.append((keyword_text, table_name, keyword_text_to_global_idx[keyword_text]))
#
#     print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
#     if not global_keywords_texts:
#         print("⛔ 분류할 키워드가 DB에 없습니다. 스크립트를 종료합니다.")
#         conn.close()
#         sys.exit(1)
#
#     global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
#     print("✔️ 모든 키워드 임베딩 생성 완료.")
#     # ----------------------------------------------------------------------------------
#
#     try:
#         # --- DB 쿼리 수정: detail 컬럼 추가 ---
#         cursor.execute("SELECT user_id, mailnum, title, detail FROM mail_info WHERE user_id = %s", (target_user_id,))
#         mails_to_process = cursor.fetchall()  # (user_id, mailNum, title, detail) 튜플 리스트
#         # ------------------------------------
#     except pymysql.Error as e:
#         print(f"메일 조회 오류: {e}")
#         conn.close()
#         sys.exit(1)
#
#     if not mails_to_process:
#         print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
#         conn.close()
#         sys.exit(0)
#
#     for mail_idx, (user_id, mailnum, title, detail) in enumerate(mails_to_process):  # detail 추가
#         print(
#             f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
#         display_title = title if len(title) < 70 else title[:67] + "..."
#         print(f"  Mail Title: '{display_title}'")
#
#         # --- 제목과 본문 결합 ---
#         # 본문(detail)이 None일 경우 빈 문자열로 처리하여 오류 방지
#         combined_mail_text = title + " " + (detail if detail else "")
#
#         # --- 제목 또는 본문에 (광고) 포함 시 ---
#         if re.search(r'\(광고\)', combined_mail_text, re.IGNORECASE):
#             classified_category = "advertisement_keywords"
#             print(f"  !!! Detected '(광고)' in combined text. Classifying as: {classified_category} !!!")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#         # ----------------------------------------------------
#
#         # --- 결합된 텍스트에서 키워드 추출 및 임베딩 ---
#         extracted_keywords_for_mail = extract_meaningful_tokens(combined_mail_text)
#
#         if not extracted_keywords_for_mail:
#             classified_category = "Unclassified (No_Keywords_Extracted_from_Combined_Text)"
#             print(f"  No meaningful keywords extracted from combined text. Classifying as: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         print(f"  Extracted keywords for mail: {extracted_keywords_for_mail}")
#
#         # 추출된 각 키워드를 임베딩
#         extracted_keyword_embeddings = model.encode(extracted_keywords_for_mail, convert_to_tensor=True)
#         # ---------------------------------------------------------
#
#         # --- 추출된 키워드 임베딩과 전역 키워드 임베딩 간의 유사도 계산 ---
#         # 결과는 [추출된_키워드_개수, 전체_키워드_개수] 형태의 텐서
#         all_similarities_matrix = util.pytorch_cos_sim(extracted_keyword_embeddings, global_keyword_embeddings)
#
#         # 모든 유사도 점수를 평탄화하여 하나의 리스트로 만들고, 각 점수와 원래 키워드/카테고리 정보 결합
#         keyword_similarity_list = []
#         for i, extracted_kw_text_from_mail in enumerate(extracted_keywords_for_mail):  # 추출된 키워드를 순회
#             for keyword_text_from_db, category_name, global_idx in global_keyword_meta_data:  # DB 키워드를 순회
#                 similarity = all_similarities_matrix[i, global_idx].item()
#                 # 유사도 기준 필터링: MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION 보다 커야 포함
#                 if similarity > MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION:
#                     # 이 유사도 점수가 어떤 '추출된 키워드' 와 어떤 'DB 키워드/카테고리' 에서 나왔는지 기록
#                     keyword_similarity_list.append(
#                         (similarity, extracted_kw_text_from_mail, keyword_text_from_db, category_name))
#         # -------------------------------------------------------------------------------------
#
#         # 유사도 기준으로 내림차순 정렬 (추출된 키워드-DB키워드 쌍에 대한 유사도)
#         keyword_similarity_list.sort(key=lambda x: x[0], reverse=True)
#
#         # 상위 N개 유사도 쌍 선택
#         top_n_similar_pairs = keyword_similarity_list[:MAX_TOP_KEYWORDS_GLOBAL]
#
#         # Top N 유사도 쌍들을 카테고리별로 그룹화하고 유사도 합계/개수 저장
#         category_scores_sum = defaultdict(float)
#         category_scores_count = defaultdict(int)
#
#         # Top N에 해당하는 (DB) 키워드의 카테고리 정보를 사용
#         for sim, _, _, category_name in top_n_similar_pairs:
#             category_scores_sum[category_name] += sim
#             category_scores_count[category_name] += 1
#
#         # 카테고리별 평균 유사도 계산
#         final_category_averages = {}
#         for category, total_sum in category_scores_sum.items():
#             final_category_averages[category] = total_sum / category_scores_count[category]
#
#         print(
#             f"  DEBUG: Top {MAX_TOP_KEYWORDS_GLOBAL} (유사도 > {MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION}) 키워드 쌍 기반 카테고리별 평균 유사도: {final_category_averages}")
#
#         best_category = None
#         highest_avg_similarity = -1.0
#
#         if not final_category_averages:  # 유효한 유사도 쌍이 없거나 Top N에 들지 못함
#             classified_category = f"Unclassified (No_Strong_Keyword_Matches_Above_{MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION})"
#             print(f"  Classification Result: {classified_category}")
#             if update_mail_category(user_id, mailnum, classified_category):
#                 print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#             continue
#
#         for category, avg_sim in final_category_averages.items():
#             if avg_sim > highest_avg_similarity:
#                 highest_avg_similarity = avg_sim
#                 best_category = category
#
#         if best_category and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
#             classified_category = best_category
#             print(
#                 f"  Classification Result: {classified_category} (Highest Avg Sim from Top {MAX_TOP_KEYWORDS_GLOBAL} : {highest_avg_similarity:.2f})")
#         else:
#             classified_category = "Unclassified"
#             # if highest_avg_similarity < FINAL_CLASSIFICATION_THRESHOLD:
#             #     classified_category += f" (Low_Avg_Sim_Top_{MAX_TOP_KEYWORDS_GLOBAL}:{highest_avg_similarity:.2f})"
#             # else:
#             #     classified_category += " (Unknown_Reason)"
#             print(f"  Classification Result: {classified_category}")
#
#         if update_mail_category(user_id, mailnum, classified_category):
#             print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category}'.")
#         else:
#             print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")
#
# cursor.close()
# conn.close()
# print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")


'''
import sys
from sentence_transformers import SentenceTransformer, util
import pymysql
import torch
from collections import defaultdict
import re
from kiwipiepy import Kiwi  # Kiwi 라이브러리 임포트

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
        charset='utf8mb4'  # 한글, 이모지 등을 위해 utf8mb4를 사용하는 것이 좋습니다.
        # 서버, DB, 테이블 인코딩도 utf8mb4로 설정되어야 합니다.
    )
    cursor = conn.cursor()

except pymysql.Error as e:
    print(f"DB 연결 오류: {e}")
    sys.exit(1)

# --- 상수 정의 ---
MAX_TOP_KEYWORDS_GLOBAL = 50  # 전체 유사도 중 상위 N개의 키워드를 선택할 개수
MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION = 0.2  # 이 유사도 이하인 키워드는 아예 고려하지 않음
FINAL_CLASSIFICATION_THRESHOLD = 0.5  # 최종 분류 임계치
# -----------------

def get_category_id_by_name(category_name):
    """
    Categories 테이블에서 category_name에 해당하는 category_id를 조회합니다.
    """
    try:
        cursor.execute("SELECT category_id FROM Categories WHERE category_name = %s", (category_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            # print(f"경고: 카테고리 이름 '{category_name}'에 해당하는 category_id를 찾을 수 없습니다.")
            return None
    except pymysql.Error as e:
        print(f"category_id 조회 오류 (Category: {category_name}): {e}")
        return None


def update_mail_category(user_id, mail_num, classified_category_name):
    """
    mail_info 테이블의 'categori' 컬럼에 분류된 카테고리 ID 또는 'Unclassified' 문자열을 업데이트합니다.
    """
    value_to_update = classified_category_name  # 기본적으로는 분류된 카테고리 이름을 사용

    if classified_category_name.startswith("Unclassified"):
        # 'Unclassified'로 분류된 경우, 문자열 'Unclassified'를 직접 저장
        value_to_update = "Unclassified"
        print(f"  Mail classified as '{classified_category_name}'. 'categori' will be set to '{value_to_update}'.")
    else:
        # 분류된 카테고리 이름 (예: 'Finance', 'Government' 등)으로 category_id 조회
        category_id_from_db = get_category_id_by_name(classified_category_name)

        if category_id_from_db is not None:
            value_to_update = category_id_from_db
        else:
            # category_id를 찾지 못했으나 'Unclassified'가 아닌 경우, 'Unclassified' 문자열을 삽입
            print(
                f"  경고: 분류된 카테고리 '{classified_category_name}'에 대한 category_id를 찾을 수 없습니다. 'categori'를 'Unclassified'로 설정합니다.")
            value_to_update = "Unclassified"

    try:
        cursor.execute(
            f"UPDATE mail_info SET categori = %s WHERE user_id = %s AND mailnum = %s",
            (value_to_update, user_id, mail_num)
        )
        conn.commit()
        return True
    except pymysql.Error as e:
        print(f"DB 업데이트 오류 (UserID: {user_id}, MailNum: {mail_num}, Value: '{value_to_update}'): {e}")
        conn.rollback()
        return False


def extract_meaningful_tokens(text):
    """
    Kiwi를 사용하여 텍스트에서 명사, 동사, 형용사 등을 추출합니다.
    불필요한 품사는 제외하고 의미 있는 키워드 위주로 추출합니다.
    """
    if not text:  # 입력 텍스트가 None이거나 비어있으면 빈 리스트 반환
        return []

    tokens = kiwi.tokenize(text)
    meaningful_words = []
    # NNG: 일반 명사, NNP: 고유 명사
    allowed_pos = ["NNG", "NNP"]
    for token in tokens:
        # 단어 길이가 1인 경우는 제외 (불필요한 조사나 어미일 가능성이 높으므로 제외)
        # 동사/형용사는 어간만 사용하도록 처리 (예: '먹다' -> '먹')
        if token.tag in allowed_pos and len(token.form) > 1:
            if token.tag.startswith("V"):  # 동사, 형용사는 어간만 추출
                # split_compound가 빈 리스트를 반환할 경우를 대비하여 조건문 추가
                split_result = kiwi.split_compound(token.form)
                if split_result:
                    meaningful_words.append(split_result[0].form)
                else:
                    meaningful_words.append(token.form)  # 분해 안되면 원형 그대로 사용
            else:
                meaningful_words.append(token.form)
    return meaningful_words


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python mail_classification.py <target_user_id>")
        print("예시: python mail_classification.py user123")
        sys.exit(1)

    target_user_id = sys.argv[1]
    print(f"--- User '{target_user_id}'의 메일을 분류하여 DB에 업데이트합니다 ---")

    # --- 모든 카테고리-소분류-키워드 데이터를 한 번에 로드 및 임베딩 ---
    # Keywords, Subcategories, Categories 테이블을 조인하여 필요한 모든 정보 로드
    global_keywords_texts = []
    # global_keyword_meta_data는 (키워드 텍스트, 해당 키워드가 속한 Categories 테이블의 category_name, global_keywords_texts 내 인덱스)
    global_keyword_meta_data = []
    keyword_text_to_global_idx = {}  # 중복 키워드 처리를 위한 맵

    print("✔️ 모든 Categories, Subcategories, Keywords 데이터를 로드 중...")
    try:
        # 세 테이블을 조인하여 '키워드 텍스트'와 해당 '대분류 카테고리 이름'을 가져옵니다.
        cursor.execute("""
            SELECT 
                k.keyword_text, 
                c.category_name 
            FROM Keywords k
            JOIN Subcategories s ON k.subcategory_id = s.subcategory_id
            JOIN Categories c ON s.category_id = c.category_id
        """)
        db_keywords_data = cursor.fetchall()  # (keyword_text, category_name) 튜플 리스트

        if not db_keywords_data:
            print("⛔ DB에 분류할 키워드 데이터가 없습니다. 스크립트를 종료합니다.")
            conn.close()
            sys.exit(1)

        for keyword_text, category_name in db_keywords_data:
            # 키워드 텍스트의 중복을 방지하고 고유한 키워드만 임베딩 리스트에 추가
            if keyword_text not in keyword_text_to_global_idx:
                keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
                global_keywords_texts.append(keyword_text)

            # 메타데이터에는 중복된 키워드가 여러 카테고리에 속할 수 있으므로 모두 추가
            # 여기서 category_name은 'Finance', 'Government'와 같이 Categories 테이블에 있는 이름입니다.
            global_keyword_meta_data.append((keyword_text, category_name, keyword_text_to_global_idx[keyword_text]))

        print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
        global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
        print("✔️ 모든 키워드 임베딩 생성 완료.")

    except pymysql.Error as e:
        print(f"키워드 데이터 로드 오류: {e}")
        conn.close()
        sys.exit(1)
    # ----------------------------------------------------------------------------------

    try:
        cursor.execute("SELECT user_id, mailnum, title, detail FROM mail_info WHERE user_id = %s", (target_user_id,))
        mails_to_process = cursor.fetchall()
    except pymysql.Error as e:
        print(f"메일 조회 오류: {e}")
        conn.close()
        sys.exit(1)

    if not mails_to_process:
        print(f"사용자 '{target_user_id}'의 메일이 없습니다.")
        conn.close()
        sys.exit(0)

    for mail_idx, (user_id, mailnum, title, detail) in enumerate(mails_to_process):
        print(
            f"\n--- {mail_idx + 1}/{len(mails_to_process)} Processing Mail (UserID: {user_id}, MailNum: {mailnum}) ---")
        display_title = title if len(title) < 70 else title[:67] + "..."
        print(f"  Mail Title: '{display_title}'")

        combined_mail_text = title + " " + (detail if detail else "")

        # --- 제목 또는 본문에 (광고) 포함 시 'Advertisement' 카테고리로 분류 ---
        # "Advertisement"는 Categories 테이블에 실제로 존재하는 category_name이어야 합니다.
        if re.search(r'\(광고\)', combined_mail_text, re.IGNORECASE):
            classified_category_name = "Advertisement"
            print(f"  !!! Detected '(광고)' in combined text. Classifying as: {classified_category_name} !!!")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
            continue
        # ---------------------------------------------------------------------

        extracted_keywords_for_mail = extract_meaningful_tokens(combined_mail_text)

        if not extracted_keywords_for_mail:
            classified_category_name = "Unclassified (No_Keywords_Extracted_from_Combined_Text)"
            print(f"  No meaningful keywords extracted from combined text. Classifying as: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
            continue

        print(f"  Extracted keywords for mail: {extracted_keywords_for_mail}")

        extracted_keyword_embeddings = model.encode(extracted_keywords_for_mail, convert_to_tensor=True)

        all_similarities_matrix = util.pytorch_cos_sim(extracted_keyword_embeddings, global_keyword_embeddings)

        keyword_similarity_list = []
        for i, extracted_kw_text_from_mail in enumerate(extracted_keywords_for_mail):
            for keyword_text_from_db, category_name, global_idx in global_keyword_meta_data:
                similarity = all_similarities_matrix[i, global_idx].item()
                if similarity > MIN_SIMILARITY_THRESHOLD_FOR_CONSIDERATION:
                    # 여기에서 category_name은 Categories 테이블의 실제 이름입니다.
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

        best_category_name = None  # Categories 테이블의 category_name이 저장될 변수
        highest_avg_similarity = -1.0

        if not final_category_averages:
            classified_category_name = "Unclassified (No_Strong_Keyword_Matches_Above_Threshold)"  # 명확한 Unclassified로 변경
            print(f"  Classification Result: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
            continue

        for category_name, avg_sim in final_category_averages.items():
            if avg_sim > highest_avg_similarity:
                highest_avg_similarity = avg_sim
                best_category_name = category_name

        if best_category_name and highest_avg_similarity >= FINAL_CLASSIFICATION_THRESHOLD:
            classified_category_name = best_category_name  # 최종 분류된 Categories 테이블의 category_name
            print(
                f"  Classification Result: {classified_category_name} (Highest Avg Sim from Top {MAX_TOP_KEYWORDS_GLOBAL} : {highest_avg_similarity:.2f})")
        else:
            classified_category_name = "Unclassified (Below_Final_Classification_Threshold)"  # 임계치 미달 시 Unclassified로 변경
            print(f"  Classification Result: {classified_category_name}")

        if update_mail_category(user_id, mailnum, classified_category_name):
            print(f"UserID {user_id}, MailNum {mailnum} updated to '{classified_category_name}'.")
        else:
            print(f"  Failed to update UserID {user_id}, MailNum {mailnum}.")

cursor.close()
conn.close()
print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
'''

'''
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

"""    Categories 테이블에서 category_name에 해당하는 category_id 조회     """
def get_category_id_by_name(category_name):

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

"""   mail_info 테이블의 'categori' 컬럼에 분류된 값을 업데이트
    classified_value는 category_id (INT) 또는 'Unclassified' (STR) 등 """
def update_mail_category(user_id, mail_num, classified_value):
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

"""명사, 고유 명사 추출""" 
def extract_meaningful_tokens(text):
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

    # --- 1차 분류를 위한 공식 메일 주소/도메인 정보 로드 ---
    # {등록된_domain_또는_full_email: category_name} 형태
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

    # --- 2차 분류를 위한 SpecialKeywords 데이터 로드 ---
    # {keyword_text: category_name} 형태
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

    print("✔️ Keywords 데이터를 로드 중...")
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
            print("⛔ DB에 분류할 키워드 데이터가 없습니다. 스크립트를 종료합니다.")
            conn.close()
            sys.exit(1)

        for keyword_text, category_name in db_keywords_data:
            if keyword_text not in keyword_text_to_global_idx:
                keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
                global_keywords_texts.append(keyword_text)
            global_keyword_meta_data.append((keyword_text, category_name, keyword_text_to_global_idx[keyword_text]))

        print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
        global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
        print("✔️ 모든 키워드 임베딩 생성 완료.")

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
        # categori가 NULL이 아니거나 'Unclassified' 문자열이 아니면 이미 분류된 것으로 간주
        if current_categori is not None and str(current_categori).strip().lower() != "unclassified":
            print(f"  Mail (MailNum: {mailnum}) already classified as '{current_categori}'. Skipping.")
            continue
        # ------------------------------------

        classified_by_any_tier = False  # 어떤 분류 단계에서든 성공했는지 플래그

        # --- 1. 발신자 이메일 주소 (sentBy)에 공식 도메인/메일 주소가 포함되는지 1차 분류 시도 ---
        if sentBy_email:
            sentBy_email_lower = sentBy_email.lower()
            for registered_domain_or_email, category_name_from_map in official_domains_map.items():
                if registered_domain_or_email in sentBy_email_lower:
                    classified_category_name = category_name_from_map
                    category_id_for_inclusion = get_category_id_by_name(classified_category_name)

                    if category_id_for_inclusion is not None:
                        if update_mail_category(user_id, mailnum, category_id_for_inclusion):
                            print(
                                f"  1차 분류 (문자열 포함): Mail classified as '{classified_category_name}' (ID: {category_id_for_inclusion}) because '{registered_domain_or_email}' is in '{sentBy_email}'.")
                            classified_by_any_tier = True
                            break  # 포함하는 첫 번째 항목을 찾으면 더 이상 확인할 필요 없음
                        else:
                            print(f"  1차 분류 실패 (문자열 포함): DB 업데이트 실패. 다음 분류 시도.")
                    else:
                        print(f"  1차 분류 (문자열 포함): 카테고리 '{classified_category_name}'에 대한 ID를 찾을 수 없습니다. 다음 분류 시도.")

            if classified_by_any_tier:  # 1차 분류에서 성공했으면 다음 메일로 넘어감
                continue
            else:
                print(f"  1차 분류 (문자열 포함): '{sentBy_email}'에 포함되는 공식 도메인/메일 주소를 찾지 못했습니다. 다음 분류 시도.")
        else:
            print(f"  1차 분류 (문자열 포함): 발신자 이메일 주소 (sentBy) 없음. 다음 분류 시도.")

        # --- 2. (1차 분류 실패 시) 메일 제목에 DB에 등록된 특별 키워드 포함 시 2차 분류 시도 ---
        # 제목이 None일 경우를 대비하여 빈 문자열로 처리
        mail_title_lower = title.lower() if title else ""

        # special_keywords_map에 있는 각 키워드가 mail_title_lower에 포함되는지 확인
        for special_keyword_text, category_name_from_map in special_keywords_map.items():
            if special_keyword_text in mail_title_lower:
                classified_category_name = category_name_from_map
                category_id_special_keyword = get_category_id_by_name(classified_category_name)

                if category_id_special_keyword is not None:
                    if update_mail_category(user_id, mailnum, category_id_special_keyword):
                        print(
                            f"  2차 분류 (특별 키워드 - 제목): Detected '{special_keyword_text}' in mail title. Classifying as '{classified_category_name}' (ID: {category_id_special_keyword}).")
                        classified_by_any_tier = True
                        break  # 해당 메일에 대해 특별 키워드를 찾았으면 더 이상 다른 특별 키워드 확인할 필요 없음
                    else:
                        print(f"  2차 분류 실패 (특별 키워드 - 제목): DB 업데이트 실패. 키워드 분류 시도.")
                else:
                    print(
                        f"  경고: 특별 키워드 '{special_keyword_text}'에 대한 카테고리 ID '{classified_category_name}'를 찾을 수 없습니다. 키워드 분류 시도.")

        if classified_by_any_tier:  # 2차 분류에서 성공했으면 다음 메일로 넘어감
            continue

        # --- 3. (1, 2차 분류 모두 실패 시) 키워드 기반 분류 실행 (기존 로직) ---
        combined_mail_text = (title if title else "") + " " + (detail if detail else "")
        extracted_keywords_for_mail = extract_meaningful_tokens(combined_mail_text)

        if not extracted_keywords_for_mail:
            classified_category_name = "Unclassified"
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
            classified_category_name = "Unclassified "
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
            classified_category_name = "Unclassified "
            print(f"  3차 분류 (키워드): Classification Result: {classified_category_name}")
            if update_mail_category(user_id, mailnum, classified_category_name):
                print(f"UserID {user_id}, MailNum {mailnum} updated to 'Unclassified'.")

cursor.close()
conn.close()
print(f"\n--- User '{target_user_id}'의 메일 분류 및 DB 업데이트 완료 ---")
'''


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
    Categories 테이블에서 category_name에 해당하는 category_id를 조회합니다.
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
    mail_info 테이블의 'categori' 컬럼에 분류된 값을 업데이트합니다.
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
    Kiwi를 사용하여 텍스트에서 명사, 고유 명사를 추출합니다.
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
    # {등록된_domain_또는_full_email: category_name} 형태
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
    # {keyword_text: category_name} 형태
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

    print("✔️ Keywords 데이터를 로드 중...")
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
            print("⛔ DB에 분류할 키워드 데이터가 없습니다. 스크립트를 종료합니다.")
            conn.close()
            sys.exit(1)

        for keyword_text, category_name in db_keywords_data:
            if keyword_text not in keyword_text_to_global_idx:
                keyword_text_to_global_idx[keyword_text] = len(global_keywords_texts)
                global_keywords_texts.append(keyword_text)
            global_keyword_meta_data.append((keyword_text, category_name, keyword_text_to_global_idx[keyword_text]))

        print(f"✔️ 총 {len(global_keywords_texts)}개의 유니크 키워드 임베딩을 생성합니다.")
        global_keyword_embeddings = model.encode(global_keywords_texts, convert_to_tensor=True)
        print("✔️ 모든 키워드 임베딩 생성 완료.")

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