"""프롬프트 최적화 모델 추론 (Qwen2.5-3B-Instruct + LoRA).

원본 프롬프트 -> 역할 명시·모호성 해소·구조화·출력 형식 명시를 적용한 개선된 프롬프트
([과제 유형] / [역할] / [과제] / [맥락 및 가정] / ([원문]) / [수행 단계] / [출력 형식]).

사용:
  python inference.py "블록체인 설명해줘"
  python inference.py --file 원본프롬프트.txt
  python inference.py                               # 대화형
  python inference.py "질문" --adapter outputs/qwen2.5-3b-promptopt/lora_adapter   # 직접 학습한 로컬 어댑터
"""
import argparse, contextlib, threading

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora"

# 학습 때와 같은 시스템 프롬프트 (바꾸면 모델 성능이 떨어진다)
SYSTEM_PROMPT = ("너는 프롬프트 최적화기다. <prompt> 안의 프롬프트에 답하지 말고, "
                 "역할 명시·모호성 해소·구조화·출력 형식 명시를 적용한 개선된 프롬프트만 출력하라.")


def build_messages(prompt):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<prompt>\n{prompt.strip()}\n</prompt>"}]


class PromptOptimizer:
    """베이스 모델 + LoRA 어댑터(Hugging Face Hub 또는 로컬 경로)."""

    def __init__(self, adapter=ADAPTER, base=BASE_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="auto")
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.lock = threading.Lock()  # GPU 하나 + 어댑터 on/off 상태 공유: 생성은 한 번에 하나씩

    def _generate_kwargs(self, prompt, temperature, repetition_penalty, max_new_tokens):
        enc = self.tok.apply_chat_template(build_messages(prompt), add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(self.model.device)
        kw = dict(**enc, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                  do_sample=temperature > 0)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=0.9)
        return kw

    def _run(self, kw, use_adapter):
        # use_adapter=False: 어댑터를 끈 베이스 모델 (파인튜닝 전과 비교용)
        ctx = contextlib.nullcontext() if use_adapter else self.model.disable_adapter()
        with self.lock, ctx, self.torch.inference_mode():
            return self.model.generate(**kw)

    def __call__(self, prompt, temperature=0.0, repetition_penalty=1.1, max_new_tokens=1024, use_adapter=True):
        """원본 프롬프트 -> 개선된 프롬프트 (문자열)."""
        kw = self._generate_kwargs(prompt, temperature, repetition_penalty, max_new_tokens)
        out = self._run(kw, use_adapter)
        return self.tok.decode(out[0, kw["input_ids"].shape[1]:], skip_special_tokens=True)

    def stream(self, prompt, temperature=0.0, repetition_penalty=1.1, max_new_tokens=1024, use_adapter=True):
        """생성되는 대로 텍스트 조각을 내보내는 이터레이터."""
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = self._generate_kwargs(prompt, temperature, repetition_penalty, max_new_tokens)
        threading.Thread(target=self._run, args=({**kw, "streamer": streamer}, use_adapter), daemon=True).start()
        return streamer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--adapter", default=ADAPTER, help="Hugging Face 저장소 ID 또는 로컬 어댑터 경로")
    ap.add_argument("--file", help="원본 프롬프트가 담긴 텍스트 파일")
    a = ap.parse_args()

    opt = PromptOptimizer(a.adapter)

    def show(p):
        print("=" * 70)
        for piece in opt.stream(p):
            print(piece, end="", flush=True)
        print()

    if a.file:
        show(open(a.file, encoding="utf-8").read())
    elif a.prompt:
        show(" ".join(a.prompt))
    else:  # 인자 없으면 대화형
        print("원본 프롬프트를 입력하세요 (빈 줄 입력 시 종료)")
        while (q := input("> ").strip()):
            show(q)
