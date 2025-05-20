from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper, faiss, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from pydub import AudioSegment
import os

app = FastAPI(
    title="Voice Phishing Detection API",
    description="이 API는 업로드된 음성에서 보이스피싱 문장을 탐지합니다.",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 인덱스 로드
whisper_model = whisper.load_model("medium")  # base → medium 업그레이드
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("phishing_index.faiss")

with open("phishing_texts.pkl", "rb") as f:
    phishing_texts = pickle.load(f)

# 결과 모델
class AnalyzeResult(BaseModel):
    text: str
    similarity_score: float
    phishing_detected: bool
    matched_example: str


@app.post("/analyze-audio/", response_model=AnalyzeResult, tags=["Voice Phishing Detection"])
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 파일 확장자 확인 및 로드
        filename = file.filename
        extension = os.path.splitext(filename)[-1].replace(".", "")
        audio = AudioSegment.from_file(file.file, format=extension)

        # 🔹 음질 개선: 16kHz, mono 설정
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export("temp_audio.wav", format="wav")

        # 🔹 한국어로 고정
        result = whisper_model.transcribe("temp_audio.wav", language='ko')
        text = result["text"]

        # 문장 임베딩 및 유사도 검색
        embedding = embed_model.encode([text], convert_to_numpy=True)
        D, I = index.search(embedding, k=1)
        matched_embedding = index.reconstruct(int(I[0][0]))

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarity = cosine_similarity(embedding[0], matched_embedding)
        threshold = 0.7
        phishing_detected = similarity > threshold

        return AnalyzeResult(
            text=text,
            similarity_score=float(similarity),
            phishing_detected=phishing_detected,
            matched_example=phishing_texts[I[0][0]]
        )

    except Exception as e:
        return {"error": f"오류 발생: {str(e)}"}