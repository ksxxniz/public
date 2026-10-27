import ollama

# 설치된 모델 목록 출력
for m in ollama.list().models:
    print("설치됨:", m.model)

# EXAONE에게 간단히 말 걸어보기
resp = ollama.chat(
    model="exaone3.5:7.8b-instruct-q8_0",   # ollama list에 나온 이름으로 수정
    messages=[{"role": "user", "content": "안녕? 한 문장으로 자기소개 해줘."}],
    options={"temperature": 0},
)
print(resp["message"]["content"])
print("읽은 토큰 수:", resp["prompt_eval_count"])