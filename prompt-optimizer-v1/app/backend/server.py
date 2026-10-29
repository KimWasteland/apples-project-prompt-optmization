"""프롬프트 최적화 로컬 서버: 모델 API + 웹 화면 (http://localhost:8000).

실행 (저장소 루트에서):
  cd app/frontend && npm install && npm run build && cd ../..   # 웹 화면 빌드 (처음 한 번)
  python app/backend/server.py
환경변수:
  ADAPTER  Hugging Face 저장소 ID 또는 로컬 어댑터 경로 (기본: inference.ADAPTER)
"""
import os, sys, threading, time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # 저장소 루트의 inference.py
from inference import ADAPTER as DEFAULT_ADAPTER, PromptOptimizer

ADAPTER = os.environ.get("ADAPTER", DEFAULT_ADAPTER)
WEB = ROOT / "app" / "frontend" / "out"  # npm run build 결과

optimizer = PromptOptimizer(ADAPTER)
gpu = threading.Lock()  # GPU 1장: 요청을 순서대로 처리

app = FastAPI(title="Prompt Optimizer")


class OptimizeIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.post("/optimize")
def optimize(body: OptimizeIn):
    t0 = time.time()
    with gpu:
        r = optimizer(body.text)
    return {"prompt": r["prompt"], "label": r["label"], "fixes": r["fixes"],
            "request": r["request"], "has_content": bool(r["content"]),
            "latency": {"total": round(time.time() - t0, 2)}}


if WEB.is_dir():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
else:
    @app.get("/")
    def no_web():
        return PlainTextResponse("웹 화면이 빌드되지 않았습니다: cd app/frontend && npm install && npm run build")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
