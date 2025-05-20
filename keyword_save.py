import zipfile
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


# 1. 텍스트 파일 수집
text_files = glob.glob("/mnt/data/extracted_txt/**/*.txt", recursive=True)

# 2. 텍스트 내용 수집
extracted_sentences = []
for file_path in text_files:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        # 문장 단위로 나누기 (더 고급처리를 원하면 nltk 등으로 분절 가능)
        sentences = content.split('\n')
        extracted_sentences.extend([s.strip() for s in sentences if s.strip()])

# 3. 기존 보이스피싱 데이터
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

# 4. TXT 파일에서 읽은 문장 추가 (또는 조건 필터링하여 추림 가능)
phishing_data.extend(extracted_sentences)

# 5. 문장 임베딩
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(phishing_data, convert_to_numpy=True)

# 6. FAISS 인덱스 생성 및 저장
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))

# 저장
faiss.write_index(index, "phishing_index_updated.faiss")
with open("phishing_texts_updated.pkl", "wb") as f:
    pickle.dump(phishing_data, f)