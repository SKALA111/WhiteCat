from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper, faiss, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from pydub import AudioSegment
app = FastAPI()
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
@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    # 1. 오디오 저장
    audio = AudioSegment.from_file(file.file)
    audio.export("temp_audio.wav", format="wav")
    # 2. 음성 → 텍스트
    result = whisper_model.transcribe("temp_audio.wav")
    text = result["text"]
    
    # 3. 유사도 측정
    embedding = embed_model.encode([text], convert_to_numpy=True)
    D, I = index.search(embedding, k=1)  # 가장 가까운 문장 1개
    similarity = D[0][0]  # L2 거리 (작을수록 유사)
    phishing_detected = similarity < 1.0  # 임계값 조절 가능
    return {
        "text": text,
        "similarity_score": float(similarity),
        "phishing_detected": phishing_detected,
        "matched_example": phishing_texts[I[0][0]]
    }