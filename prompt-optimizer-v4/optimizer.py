"""v4 프롬프트 최적화 모델 추론과 v4 규칙.

사용:
  python prompt-optimizer-v4/optimizer.py "초생달이 뭐야"

v4 규칙 두 가지
  gate()     다시 쓸 질문인지 고른다 (짧고 조건이 없는 질문만)
  faithful() 모델이 다시 쓴 글에 문제가 없는지 검사한다
이 규칙은 추론할 때와 v4 학습 데이터를 만들 때(build_data.py) 똑같이 쓴다.
"""
import difflib
import re
import sys

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = ("너는 프롬프트 최적화기다. <prompt> 안의 요청에 답하지 말고, 원래 요청의 정보와 말투를 모두 살린 채 "
          "좋은 답에 필요한 내용을 함께 부탁하는 한 문단의 프롬프트로 다시 써라. 정답이 하나로 정해진 요청은 원래 요청만 출력하라.")

# 학습된 어댑터. Hugging Face에서 내려받는다. 직접 학습한 어댑터를 쓰려면 "prompt-optimizer-v4/lora_adapter"로 바꾼다.
ADAPTER = "Wasteland0103/qwen2.5-3b-prompt-optimizer-v4-lora"


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


def load_optimizer():
    """'질문 -> 답변 모델에 보낼 프롬프트' 함수를 돌려준다."""
    model = PromptOptimizer()

    def optimize(question):
        if not gate(question):
            return question   # 다시 쓸 대상이 아니면 모델을 부르지 않는다
        # repetition_penalty 1.1은 원문을 옮겨 쓰는 것까지 막아서 1.0을 쓴다
        output = model(question, repetition_penalty=1.0)
        return final(question, output)

    return optimize


# ---------- v4 규칙 1: gate (다시 쓸 질문인가) ----------
MAX_CHARS = 60   # v3 데이터에서 60자 이하 질문에서만 이득이 있었다

# 분량 조건. 예: "3줄", "다섯 문장", "10~20자"
QUANTITY = re.compile(r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:~\s*\d+\s*)?"
                      r"(자|글자|줄|문장|단어|개|단락|문단|쪽|페이지|장|가지|행|곳|편)")
# 형식 조건. 예: "표로", "영어로", "간단히"
FORMAT = re.compile(r"표로|목록으로|리스트로|JSON|json|한 문장|한 줄|영어로|일본어로|중국어로|번역|쉼표|따옴표|대문자|소문자|"
                    r"형식으로|양식으로|마크다운|글머리|분량|A4|이내로|이내의|간단히|간단하게|짧게|간략|핵심만")
# 수식. 예: "3+5", "2 x 4 = 8"의 "2 = 8"
MATH = re.compile(r"\d\s*[+\-*/×÷^=]\s*\d")
# 코드
CODE = re.compile(r"```|def |import |SELECT |select |[{};]")


def gate(question):
    """60자 이하 한 줄이고, 분량, 형식, 수식, 코드가 없는 질문만 True"""
    q = question.strip()
    if len(q) > MAX_CHARS:
        return False
    if "\n" in q:
        return False
    if QUANTITY.search(q) or FORMAT.search(q) or MATH.search(q) or CODE.search(q):
        return False
    return True


# ---------- v4 규칙 2: faithful (다시 쓴 글 검사) ----------
SENTENCE_END = r"[.?!~ㅠㅜㅎㅋ^\s]*$"   # 문장 끝의 문장부호, 이모티콘
POLITE_END = re.compile(r"(요|니다|세요|까요|시오|죠|여)" + SENTENCE_END)                                  # 존댓말 끝
CASUAL_END = re.compile(r"(줘|해|야|어|아|냐|니|지|래|까|임|음|됨|함|라|봐|자|ㄱ|plz|좀)" + SENTENCE_END)   # 반말 끝
CJK = re.compile(r"[一-鿿぀-ヿ]")   # 한자, 일본 글자


def sentences(text):
    """마침표, 물음표, 느낌표 뒤에서 문장을 나눈다."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [part for part in parts if part]


def is_polite(sentence):
    return bool(POLITE_END.search(sentence))


def tone_of(question):
    """반말로 끝나면 'casual'. 존댓말이나 명사로 끝나면 'polite'."""
    q = question.strip()
    if CASUAL_END.search(q) and not POLITE_END.search(q):
        return "casual"
    return "polite"


def coverage(original, output):
    """원래 질문의 글자가 출력에 순서대로 몇 % 남아 있는가 (공백은 무시)"""
    a = re.sub(r"\s", "", original)
    b = re.sub(r"\s", "", output)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    kept = 0
    for block in matcher.get_matching_blocks():
        kept += block.size
    return kept / max(len(a), 1)


def faithful(original, output):
    """문제 목록을 돌려준다. 빈 목록이면 통과."""
    if output.strip() == original.strip():
        return []   # 원문 그대로면 문제없다

    problems = []

    # 한 문단이어야 한다. 줄바꿈, 괄호, 가이드 흔적이 있으면 안 된다
    if "\n" in output.strip() or re.search(r"[\[\]()<>{}]|다룰 내용|갈림길", output):
        problems.append("format")

    # 원문에 없던 한자, 일본 글자가 섞이면 안 된다
    if CJK.search(output) and not CJK.search(original):
        problems.append("cjk")

    # 1~4문장이어야 한다
    output_sentences = sentences(output)
    if len(output_sentences) < 1 or len(output_sentences) > 4:
        problems.append("sentences")

    # 말투가 원문과 같아야 한다
    polite = [is_polite(s) for s in output_sentences]
    if tone_of(original) == "polite" and not all(polite):
        problems.append("tone")
    if tone_of(original) == "casual" and any(polite):
        problems.append("tone")

    # 원문 글자가 80% 이상 남아 있어야 한다
    if coverage(original, output) < 0.8:
        problems.append("info_loss")

    # 원문의 영어 용어와 숫자는 글자 그대로 남아야 한다
    terms = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*|\d+", original)
    for term in terms:
        if term not in output:
            problems.append("term_lost")
            break

    # 원문보다 170자 넘게 길어지면 안 된다
    if len(output) > len(original) + 170:
        problems.append("too_long")

    return problems


def final(original, output):
    """다시 쓴 글이 검사를 통과하면 그것을, 아니면 원래 질문을 보낸다."""
    output = output.strip()
    if not gate(original):
        return original
    if output == "":
        return original
    if faithful(original, output):
        return original
    return output


if __name__ == "__main__":
    optimize = load_optimizer()
    print(optimize(" ".join(sys.argv[1:])))
