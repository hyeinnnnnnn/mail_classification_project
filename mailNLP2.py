
import pymysql
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from kiwipiepy import Kiwi


# 사전학습된 KoBERT 모델 불러오기
kiwi = Kiwi()
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


# db연결
mydb = pymysql.connect(
    host='10.50.131.18',
    port = 3306,
    user='user',
    password='user1234',
    database='maildb',
)



def extract_keywords(text):
    return [token.form for token in kiwi.tokenize(text) if token.tag in ['NNG', 'NNP']]



# 쿼리 실행


mycursor = mydb.cursor()


mycursor.execute(f"SELECT detail FROM mail_info where mailNum = 8 ")

# 결과 가져오기
result = mycursor.fetchall()

# 결과 출력
for row in result:
    mail_detail = row[0]
    print(row)

# SBERT 모델 불러오기
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


def extract_keywords(text):
    return [token.form for token in kiwi.tokenize(text) if token.tag in ['NNG', 'NNP']]

def classify_by_keywords(mail_text: str, category_keywords: dict, top_k=100) -> str:
    # 메일에서 키워드 추출
    mail_keywords = extract_keywords(mail_text)
    if not mail_keywords:
        return "분류 불가 (키워드 없음)"

    mail_vecs = model.encode(mail_keywords)

    category_scores = {}

    for category, keywords in category_keywords.items():
        category_vecs = model.encode(keywords)
        # mail 키워드와 카테고리 키워드 간 모든 유사도 계산
        sim_matrix = cosine_similarity(mail_vecs, category_vecs)
        top_sims = np.sort(sim_matrix.flatten())[-top_k:]  # 가장 유사한 top_k
        avg_score = np.mean(top_sims)
        category_scores[category] = avg_score

    for cat, score in category_scores.items():
        print(f"{cat}: 평균 유사도 {score:.4f}")

    best = max(category_scores, key=category_scores.get)
    return best


result = classify_by_keywords(mail_detail, category_keywords)
print(f"\n최종 분류: {result}")

mydb.close()
if __name__ == '__main__':
    app.run()