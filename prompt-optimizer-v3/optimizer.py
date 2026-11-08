"""v3 프롬프트 최적화 모델 추론.

사용:
  python prompt-optimizer-v3/optimizer.py "참고문헌 관리 프로그램 비교해주세요"
"""
import sys

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# v2와 같은 시스템 프롬프트로 학습했다
SYSTEM = ("너는 프롬프트 최적화기다. <prompt> 안의 프롬프트에 답하지 말고, "
          "역할 명시·모호성 해소·구조화·출력 형식 명시를 적용한 개선된 프롬프트만 출력하라.")

# 학습된 어댑터. Hugging Face에서 내려받는다. 직접 학습한 어댑터를 쓰려면 "prompt-optimizer-v3/lora_adapter"로 바꾼다.
ADAPTER = "Wasteland0103/qwen2.5-3b-prompt-optimizer-v3-lora"
V2_ADAPTER = "Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora"   # 설계 실험(experiments.py)에서 v2 출력을 만들 때 쓴다


# ---------- 모델 ----------
def build_messages(question):
    """모델에 넣는 대화. 학습과 추론에서 똑같은 형식을 쓴다."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"<prompt>\n{question.strip()}\n</prompt>"},
    ]


class PromptOptimizer:
    """베이스 모델에 LoRA 어댑터를 붙여 올린다. optimizer(질문)을 부르면 개선된 프롬프트를 돌려준다."""

    def __init__(self, adapter=ADAPTER):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()

    def __call__(self, question, repetition_penalty=1.1, max_new_tokens=1024):
        messages = build_messages(question)
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, repetition_penalty=repetition_penalty,
                                         max_new_tokens=max_new_tokens)
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_length:]   # 입력 뒤에 새로 만든 부분만
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    optimize = PromptOptimizer()
    print(optimize(" ".join(sys.argv[1:])))
