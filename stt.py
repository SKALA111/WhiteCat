import os
import whisper
import re

# 음성 파일이 저장된 폴더 경로
AUDIO_DIR = "./samples"
# 변환된 텍스트 파일을 저장할 폴더 경로
TEXT_DIR = "./samples_scripts"

# Whisper 모델 불러오기 ("base", "small", "medium", "large" 중 선택 가능)
model = whisper.load_model("base")

# samples 폴더 안의 모든 mp3 파일을 처리
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".mp3"):
        # mp3 파일 경로
        filepath = os.path.join(AUDIO_DIR, filename)
        # Whisper로 STT 변환
        result = model.transcribe(filepath)
        
        # 변환된 텍스트
        raw_text = result["text"]

        # 저장할 텍스트 파일 경로 설정 (.mp3 → .txt)
        txt_name = filename.replace(".mp3", ".txt")
        txt_path = os.path.join(TEXT_DIR, txt_name)
        
        # 텍스트 파일로 저장 (UTF-8 인코딩)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        print(f"[완료] {filename} → {txt_name}")
