from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper, faiss, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment
import os

app = FastAPI(
    title="Voice Phishing Detection API",
    description="이 API는 업로드된 음성에서 보이스피싱 문장을 탐지합니다.",
    version="1.0.0"
)
# CORS 설정 (React와 연동 시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 모델 & 인덱스 로드
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("phishing_index.faiss")

with open("phishing_texts.pkl", "rb") as f:
    phishing_texts = pickle.load(f)

# 결과 모델 (Swagger 문서용)
class AnalyzeResult(BaseModel):
    text: str
    similarity_score: float
    phishing_detected: bool
    matched_example: str


@app.post("/analyze-audio/", response_model=AnalyzeResult, tags=["Voice Phishing Detection"])
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 파일 확장자 파악
        filename = file.filename
        extension = os.path.splitext(filename)[-1].replace(".", "")
        audio = AudioSegment.from_file(file.file, format=extension)

        # wav로 임시 저장
        audio.export("temp_audio.wav", format="wav")

        # 음성 → 텍스트 변환
        result = whisper_model.transcribe("temp_audio.wav")
        text = result["text"]

        # 문장 임베딩 후 FAISS 검색
        embedding = embed_model.encode([text], convert_to_numpy=True)

        # 인덱스에서 가장 가까운 벡터 찾기 (L2 거리 기반)
        D, I = index.search(embedding, k=1)  # 가장 가까운 문장 1개
        matched_embedding = index.reconstruct(int(I[0][0]))

        # 코사인 유사도 함수 정의
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarity = cosine_similarity(embedding[0], matched_embedding)

        threshold = 0.7  # 코사인 유사도 기준값, 필요에 따라 조절하세요
        phishing_detected = similarity > threshold

        return AnalyzeResult(
            text=text,
            similarity_score=float(similarity),
            phishing_detected=phishing_detected,
            matched_example=phishing_texts[I[0][0]]
        )

    except Exception as e:
        return {
            "error": f"오류 발생: {str(e)}"
        }