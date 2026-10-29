"""프롬프트 최적화 모델 데모 서버.

실행 (conda test 환경, 프로젝트 루트에서):
    python app/server.py            # http://127.0.0.1:7860
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401
from unsloth import FastLanguageModel

import contextlib
import json
import sys
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", ROOT / "outputs" / "qwen2.5-3b-promptopt" / "lora_adapter"))
SYSTEM_PROMPT = json.loads((ROOT / "config" / "quota.json").read_text(encoding="utf-8"))["system_prompt"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR), max_seq_length=2048, load_in_4bit=True, dtype=None
)
FastLanguageModel.for_inference(model)
gen_lock = threading.Lock()  # GPU 하나라 요청은 한 번에 하나씩
MAX_PROMPT_CHARS = 3000  # 외부 공개 시 과도한 입력 방지

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
    rows = [json.loads(line) for line in open(ROOT / "data" / "final" / "sft_test.jsonl", encoding="utf-8")]
    out = []
    for r in rows[:12]:
        text = r["messages"][1]["content"].removeprefix("<prompt>\n").removesuffix("\n</prompt>")
        out.append(text)
    return out


@app.post("/generate")
def generate(req: GenRequest):
    if not req.prompt.strip() or len(req.prompt) > MAX_PROMPT_CHARS:
        return PlainTextResponse(f"입력은 1~{MAX_PROMPT_CHARS}자여야 합니다.", status_code=400)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<prompt>\n{req.prompt.strip()}\n</prompt>"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=min(req.max_new_tokens, 1536),
        repetition_penalty=req.repetition_penalty,
        do_sample=req.temperature > 0,
    )
    if req.temperature > 0:
        kwargs.update(temperature=req.temperature, top_p=0.9)

    def run():
        with gen_lock, torch.inference_mode():
            ctx = contextlib.nullcontext() if req.use_adapter else model.disable_adapter()
            with ctx:
                model.generate(**kwargs)

    threading.Thread(target=run, daemon=True).start()
    return StreamingResponse(streamer, media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 7860)))
