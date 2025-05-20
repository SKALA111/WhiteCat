import os

# 텍스트 줄바꿈 함수 (고정 길이 기준, 줄바꿈 강제)
def insert_line_breaks(text, max_len=60):
    lines = []
    while len(text) > max_len:
        split_point = text.rfind(' ', 0, max_len)
        if split_point == -1:
            split_point = max_len
        lines.append(text[:split_point].strip())
        text = text[split_point:].strip()
    lines.append(text)
    return '\n'.join(lines)

# 변환된 텍스트가 저장된 폴더 (줄바꿈도 여기에 덮어쓰기)
TEXT_DIR = "./samples_scripts"

# 모든 .txt 파일에 줄바꿈 적용 후 덮어쓰기
for filename in os.listdir(TEXT_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(TEXT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 줄바꿈 적용
        cleaned_text = insert_line_breaks(raw_text)

        # 같은 경로에 덮어쓰기 저장
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"[줄바꿈 적용 완료] {filename} (덮어쓰기)")
