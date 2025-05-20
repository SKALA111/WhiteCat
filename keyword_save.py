import zipfile
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# 1. 모델 로딩
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. 기존 보이스피싱 키워드 및 시나리오
phishing_data = [
    "계좌 이체", "보증금", "송금", "즉시 입금", "대출 승인", "통장 대여",
    "검찰청", "금융감독원", "국세청", "은행 본점", "통신사 고객센터", "정부지원금",
    "긴급", "즉시 조치", "출석 요구", "체포영장", "법적 조치", "소환장", "압수 수색", "벌금",
    "주민등록번호", "공인인증서", "OTP 번호", "계좌번호", "보안카드", "비밀번호", "카드번호", "인증번호",
    "귀하의 계좌가 범죄에 연루되었습니다", "즉시 연락 바랍니다", "안내받은 계좌로 이체해주세요",
    "수사 협조를 부탁드립니다", "정부 지원금 신청 대상입니다",
    "검찰청입니다. 귀하의 명의로 대포통장이 개설되었습니다",
    "금융감독원 조사관입니다. 안전한 계좌로 자산을 옮기셔야 합니다",
    "범죄 연루 혐의로 계좌가 동결 예정입니다. 본인 확인이 필요합니다"
]

# 3. samples 폴더 내 모든 sample59.txt ~ sample80.txt 읽기
sample_folder = "./samples"
for i in range(1, 1000):
    filename = f"sample{i}.txt"
    path = os.path.join(sample_folder, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  # 내용이 있는 경우만 추가
                phishing_data.append(content)

# 4. 임베딩 처리
embeddings = model.encode(phishing_data, convert_to_numpy=True)

# 5. FAISS 인덱스 생성
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# 6. 원문 저장 (pickle)
with open("phishing_texts.pkl", "wb") as f:
    pickle.dump(phishing_data, f)

# 7. FAISS 인덱스 저장
faiss.write_index(index, "phishing_index.faiss")
