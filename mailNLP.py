from flask import Flask, request, jsonify
import mysql.connector
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


app = Flask(__name__)

# 사전학습된 KoBERT 모델 불러오기
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertModel.from_pretrained('monologg/kobert')

"""
    
"""
# db연결
mydb = mysql.connector.connect(
    host='localhost',
    user='test',
    password='2370',
    database='textdb'
)

"""
    변수명 
    category_keyword(dict/list) -> 키워드와 세분화 DB로 불러오기.
    sentens -> mailDetail
"""
# 나중에 db로 받아와야 함
# dictionary(key(keyword), list(keywordDetail))로 받아올 것

category_keywords = {
    "금융": [
        "카드", "신용카드", "체크카드", "금융", "은행", "명세서", "사용 내역", "거래 내역",
        "대출", "대출금", "이자", "상환", "연체", "계좌", "잔액", "이체", "출금", "입금",
        "자동이체", "청구서", "납부", "수수료", "결제", "카드 승인", "결제 알림",
        "한도 초과", "계좌이체", "통장"
    ],
    "회사": [
        "회사", "인사팀", "인사 발령", "인사 이동", "인사 공지", "인사 평가", "인사 기록",
        "업무", "회의", "프로젝트", "계약서", "발주", "클레임", "결제요청", "지출 결의서",
        "보고서", "업무일지", "전결", "협조 요청", "출근", "퇴근", "야근", "회의록", "근태",
        "급여", "출장", "교육 안내", "사내 공지", "조직도", "부서 이동"
    ],
    "정부": [
        "국세청", "세금", "연말정산", "소득 확인", "환급", "건강보험", "국민연금", "고용보험",
        "주민센터", "정부24", "민원24", "인증서", "공공기관", "보조금", "고지서", "납부기한",
        "세무서", "전자세금계산서", "납세자", "소득공제", "의료비", "행정처리", "질병청",
        "전입신고", "주민등록", "본인 인증"
    ],
    "포털": [
        "로그인 알림", "로그인 시도", "로그인 기록", "계정 활동", "계정 잠김", "이중인증",
        "인증 코드", "OTP", "IP 변경", "보안 경고", "비밀번호 변경 요청", "비밀번호 초기화",
        "계정 도용", "비정상 접속", "사용자 인증", "접속 알림", "로그인 실패", "보안 설정",
        "로그인 기기", "새로운 기기 로그인", "이메일 인증", "인증 메일", "로그인 확인",
        "인증 실패", "앱 접근 알림", "인증 문자", "자동 로그아웃"]
}

# 연결
if mydb.is_connected():
    print("데이터베이스에 성공적으로 연결되었습니다.")
    # 쿼리 실행
    userID = ""

    """
    userID = html에서 직접 받아와야
    javascript에서 보내줘야 받아올 수 있음
    mycursor.execute(f"SELECT mail_detail FROM mail where mail_num = 1  && userID = {userID}")

    """
    mycursor = mydb.cursor()

#     sql = "INSERT INTO mail (mail_num, mail_detail) VALUES (%s, %s)"
#     val = ("9", """“”“）
    mycursor.execute(f"SELECT mail_detail FROM mail where mail_num = 9 ")

    # 결과 가져오기
    result = mycursor.fetchall()

    # 결과 출력
    for row in result:
        mail_detail = row[0]
        print(row)

# SBERT 모델 불러오기
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

@app.route('/')
def classify_sentence(sentence: str, category_keywords: dict) -> str:
    sentence_vec = model.encode([sentence])  # 2D 배열 (1, 768)

    # scores = {}
    # for category, keywords in category_keywords.items():
    #     keyword_vecs = model.encode(keywords)  # shape: (len(keywords), 768)
        #평균
    #     sim = cosine_similarity(sentence_vec, keyword_vecs).mean()
    #     scores[category] = sim
    #
    # # 유사도 출력 지우고 best_match만 보내면 됨
    # for cat, score in scores.items():
    #     print(f"{cat}: {score:.4f}")
    #
    # best_match = max(scores, key=scores.get)
    # return best_match

    all_keywords = []
    keyword_to_category = []

    for category, keywords in category_keywords.items():
        all_keywords.extend(keywords)
        keyword_to_category.extend([category] * len(keywords))

    # 전체 키워드 임베딩
    keyword_vecs = model.encode(all_keywords)  # shape: (N, 768)
    sims = cosine_similarity(sentence_vec, keyword_vecs)[0]  # shape: (N, )

    # 상위 10개 키워드 인덱스
    top_k = min(10, len(sims))
    top_indices = np.argsort(sims)[-top_k:][::-1]

    print("\n[Top 10 유사 키워드]")
    category_score_map = {}

    for i, idx in enumerate(top_indices, 1):
        keyword = all_keywords[idx]
        category = keyword_to_category[idx]
        sim = sims[idx]
        print(f"{i}. ({category}) {keyword} : {sim:.4f}")

        if category not in category_score_map:
            category_score_map[category] = []
        category_score_map[category].append(sim)

    print("\n📊 [카테고리별 평균 유사도]")
    avg_scores = {}
    for cat, score_list in category_score_map.items():
        avg = np.mean(score_list)
        avg_scores[cat] = avg
        print(f"- {cat}: 평균 {avg:.4f} ({len(score_list)}개 키워드 포함)")

    best_match = max(avg_scores, key=avg_scores.get)
    return best_match

# 테스트
# best_match -> DB - mail - keyword
#
sentence = mail_detail
category = classify_sentence(sentence, category_keywords)

print(f"\n문장 분류 결과: {category}")

if __name__ == '__main__':
    app.run()