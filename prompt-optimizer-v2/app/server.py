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
sys.path.insert(0, str(ROOT))  # 저장소 루트의 inference.py
from inference import ADAPTER as DEFAULT_ADAPTER, PromptOptimizer

optimizer = PromptOptimizer(os.environ.get("ADAPTER", DEFAULT_ADAPTER))
MAX_PROMPT_CHARS = 3000  # 과도한 입력 방지

app = FastAPI()


class GenRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    use_adapter: bool = True


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/examples")
def examples():
    rows = [json.loads(line) for line in open(ROOT / "data" / "dataset.jsonl", encoding="utf-8")]
    return [r["input"] for r in rows if r["split"] == "test"][:12]


@app.post("/generate")
def generate(req: GenRequest):
    if not req.prompt.strip() or len(req.prompt) > MAX_PROMPT_CHARS:
        return PlainTextResponse(f"입력은 1~{MAX_PROMPT_CHARS}자여야 합니다.", status_code=400)
    streamer = optimizer.stream(req.prompt, temperature=req.temperature, repetition_penalty=req.repetition_penalty,
                                max_new_tokens=min(req.max_new_tokens, 1536), use_adapter=req.use_adapter)
    return StreamingResponse(streamer, media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
