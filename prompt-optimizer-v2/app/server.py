"""프롬프트 최적화 로컬 데모 서버: 모델 API + 웹 화면.

실행 (저장소 루트에서):
    python app/server.py            # http://127.0.0.1:7860
환경변수:
    ADAPTER  Hugging Face 저장소 ID 또는 로컬 어댑터 경로 (기본: inference.ADAPTER)
"""
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
# 현재 파일의 경로 앞에 루트 파일 경로를 삽입함 > 모듈 참조시 prompt-optimizer-v2 폴더를 기준으로 탐색하게 됨
sys.path.insert(0, str(ROOT))  # 저장소 루트의 inference.py
from inference import ADAPTER as DEFAULT_ADAPTER, PromptOptimizer

optimizer = PromptOptimizer(os.environ.get("ADAPTER", DEFAULT_ADAPTER))
MAX_PROMPT_CHARS = 3000  # 과도한 입력 방지

# FastAPI의 객체 생성
app = FastAPI()

# API 요청 데이터의 구조와 기본값을 정의하는 요청 모델 구조 선언
# 어떤 요청이 들어오는지 데이터 타입을 명시한 것
# 파이썬은 동적 타이핑 언어라 데이터타입 강제에 어렵지만 Pydantic의 BaseModel을 사용하면 자동으로 타입 변환과 검증을 시도함
class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    use_adapter: bool = True


# 사용자가 도메인 루트로 GET 방식의 요청을 보냈을 때 index를 실행하라는 라우팅 데코레이터
# 브라우저 주소창에 도메인 루트로 요청을 보냈을 때 응답함
@app.get("/")
def index():
    #html 파일을 클라이언트에게 전송함
    return FileResponse(Path(__file__).parent / "index.html")

# 사용자가 /example 경로로 GET 방식의 요청을 보냈을 때 실행
@app.get("/examples")
def examples():
    rows = [json.loads(line) for line in open(ROOT / "data" / "dataset.jsonl", encoding="utf-8")]
    return [r["input"] for r in rows if r["split"] == "test"][:12]

# 클라이언트가 데이터를 담아 보낼 수 있는 POST 방식으로 /generate 경로에 요청을 보냈을 때 실행
@app.post("/generate")
def generate(req: GenRequest):
    if not req.prompt.strip() or len(req.prompt) > MAX_PROMPT_CHARS:
        return PlainTextResponse(f"입력은 1~{MAX_PROMPT_CHARS}자여야 합니다.", status_code=400)
    # 스트리머 생성
    streamer = optimizer.stream(req.prompt, temperature=req.temperature, repetition_penalty=req.repetition_penalty,
                                max_new_tokens=min(req.max_new_tokens, 1536), use_adapter=req.use_adapter)
    # 스트리머에서 답변을 반환함
    return StreamingResponse(streamer, media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    # 출력 인코딩은 utf-8로 고정
    sys.stdout.reconfigure(encoding="utf-8")
    # 웹서버 실행 uvicorn은 비동기 ASGI 서버 (웹 클라이언트의 요청을 처리해주는 고성능 서버 규격)
    # 비동기라 함은 서버가 혼자서도 수많은 요청을 멈추지 않고 동시에 효율적으로 처리하는 것을 의미한다.
    # 로컬 호스트에 포트번호 7860번으로 서버 실행
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
