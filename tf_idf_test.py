import sys
from sentence_transformers import SentenceTransformer, util
import pymysql
import torch
from collections import defaultdict
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# --- 1. 모델 및 형태소 분석기 초기화 ---
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
kiwi = Kiwi()

# --- 2. DB 연결 설정 ---
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

# --- 3. 설정 상수 ---
# 분석할 특정 user_id를 여기에 입력하세요.
# 모든 데이터를 사용하려면 None으로 설정하세요 (예: TARGET_USER_ID = None).
TARGET_USER_ID = "chonhy12345@gmail.com"  # <-- 여기에 원하는 User ID를 입력하세요.
# 모든 사용자 데이터를 원하면 None으로 변경: TARGET_USER_ID = None

# 현재 DB에 있는 카테고리 테이블 이름을 기반으로 실제 메일 데이터의 'categori' 값 사용
TARGET_CATEGORIES = ['finance_keywords', 'government_keywords', 'portal_keywords', 'advertisement_keywords']

# TF-IDF 1차 필터링 후 각 카테고리당 남길 키워드 개수
TFIDF_TOP_N = 150  # 예시 값: 150개 (300개에서 1차로 절반 줄임)

# SBERT 2차 검증 후 각 카테고리당 최종 남길 키워드 개수
SBERT_FINAL_TOP_M = 70  # 예시 값: 70개 (TF-IDF 150개에서 최종 70개)

# Kiwi 형태소 분석 시 허용할 품사 (명사, 동사, 형용사, 부사 등)
ALLOWED_POS_TAGS = ['N', 'V', 'M', 'A']  # N: 명사, V: 동사, M: 수식언(관형사/부사), A: 형용사


# --- 4. 헬퍼 함수 ---
def extract_tokens_for_tfidf(text):
    """TF-IDF 계산을 위해 텍스트에서 토큰 추출 (품사 태깅 및 1글자 단어 제외)"""
    if not text:
        return ""
    tokens = kiwi.tokenize(text)
    meaningful_words = []
    for token in tokens:
        if token.tag[0] in ALLOWED_POS_TAGS and len(token.form) > 1:
            meaningful_words.append(token.form)
    return " ".join(meaningful_words)


def get_mail_texts_by_category(category_name, user_id=None):
    """
    DB에서 특정 카테고리의 메일 제목+본문 텍스트를 가져옴.
    user_id가 제공되면 해당 user_id의 메일만 가져옵니다.
    """
    query = f"SELECT title, detail FROM mail_info WHERE categori = %s"
    params = [category_name]

    if user_id:
        query += " AND user_id = %s"
        params.append(user_id)

    cursor.execute(query, tuple(params))

    mails = cursor.fetchall()
    combined_texts = []
    for title, detail in mails:
        combined_text = title if title else ""
        if detail:
            combined_text += " " + detail
        combined_texts.append(combined_text)
    return combined_texts


def calculate_centroid_embedding(embeddings):
    """임베딩 리스트의 평균 벡터 (Centroid) 계산"""
    if not embeddings:
        return None
    return torch.mean(torch.stack(embeddings), dim=0)


# --- 메인 로직 시작 ---
if __name__ == "__main__":
    if TARGET_USER_ID:
        print(f"--- 사용자 '{TARGET_USER_ID}'의 메일 데이터를 사용하여 키워드 분석 및 선별 시작 ---")
        user_id_for_query = TARGET_USER_ID
    else:
        print("--- 모든 메일 데이터를 사용하여 키워드 분석 및 선별 시작 ---")
        user_id_for_query = None

    optimized_keywords_by_category = {}

    for category in TARGET_CATEGORIES:
        print(f"\n[✔️ {category.upper()} 카테고리 키워드 분석 시작]")

        mail_texts = get_mail_texts_by_category(category, user_id=user_id_for_query)

        if not mail_texts:
            user_info = f" (User ID: {user_id_for_query})" if user_id_for_query else " (All Users)"
            print(f"  경고: '{category}' 카테고리{user_info}에 해당하는 메일 데이터가 없습니다. 이 카테고리는 건너뜝니다.")
            continue

        # 2. TF-IDF 계산을 위한 텍스트 전처리
        processed_texts = [extract_tokens_for_tfidf(text) for text in mail_texts]

        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
        except ValueError as e:
            print(f"  TF-IDF 계산 오류 ({category}): {e}. 해당 카테고리의 데이터가 너무 적거나 다양하지 않을 수 있습니다.")
            continue

        feature_names = tfidf_vectorizer.get_feature_names_out()

        tfidf_scores = tfidf_matrix.max(axis=0).toarray().flatten()

        keyword_scores = sorted(
            [(score, word) for word, score in zip(feature_names, tfidf_scores)],
            key=lambda x: x[0],
            reverse=True
        )

        # 3. TF-IDF 기반 1차 필터링
        filtered_by_tfidf = [word for score, word in keyword_scores if score > 0][:TFIDF_TOP_N]
        print(f"  TF-IDF 1차 필터링 완료: {len(filtered_by_tfidf)}개 키워드 선택 (상위 {TFIDF_TOP_N}개)")
        if not filtered_by_tfidf:
            print(f"  TF-IDF 필터링 후 남은 키워드가 없습니다. '{category}' 카테고리는 건너뜝니다.")
            continue

        # 4. SBERT 임베딩 및 카테고리 대표 벡터 계산
        filtered_keyword_embeddings = model.encode(filtered_by_tfidf, convert_to_tensor=True)

        category_centroid_embedding = calculate_centroid_embedding(
            [model.encode(k, convert_to_tensor=True) for k in filtered_by_tfidf]
        )
        if category_centroid_embedding is None:
            print(f"  SBERT 임베딩 오류: '{category}' 카테고리의 대표 임베딩을 계산할 수 없습니다. 건너뜝니다.")
            continue

        # 5. SBERT 유사도 기반 2차 검증
        sbert_similarities = util.pytorch_cos_sim(filtered_keyword_embeddings, category_centroid_embedding).flatten()

        final_keyword_candidates = []
        for i, word in enumerate(filtered_by_tfidf):
            similarity = sbert_similarities[i].item()
            final_keyword_candidates.append((similarity, word))

        final_keyword_candidates.sort(key=lambda x: x[0], reverse=True)

        final_selected_keywords = [word for sim, word in final_keyword_candidates[:SBERT_FINAL_TOP_M]]
        optimized_keywords_by_category[category] = final_selected_keywords

        print(f"  SBERT 2차 검증 및 최종 선별 완료: {len(final_selected_keywords)}개 키워드 선택 (상위 {SBERT_FINAL_TOP_M}개)")
        # 최종적으로 선택된 키워드 전체를 출력하도록 변경했습니다.
        # print(f"  최종 '{category}' 키워드 (일부): {final_selected_keywords[:10]}...") # 이 줄이 삭제됩니다.

    print("\n--- 모든 카테고리 키워드 분석 완료 ---")
    print("\n=== 최종 최적화된 키워드 리스트 ===")
    for category, keywords in optimized_keywords_by_category.items():
        print(f"\n# {category.upper()} 키워드 ({len(keywords)}개)")
        # 여기에서 리스트 전체를 출력합니다.
        print(keywords)

    cursor.close()
    conn.close()
    print("\nDB 연결 종료.")