"""
v3 폴더의 스크립트가 같이 쓰는 코드.

1. 경로
2. 파일 읽기, 쓰기
3. 통계와 표
4. 외부 모델 호출 (GPT, Ollama 답변 모델, 보상 모델)
"""
import difflib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats


# ---------- 1. 경로 ----------
HERE = Path(__file__).resolve().parent      # 이 폴더 (prompt-optimizer-v3)
DATA = HERE / "data"                         # 학습 데이터와 데이터를 만드는 코드
EVAL = HERE / "eval"                         # 평가 코드
ROOT = HERE.parent                          # 저장소 루트
RESULTS = ROOT / "results" / "v3"           # v3 평가와 설계 실험 결과
DOCS = ROOT / "docs"                        # 문서, 결과표
BENCHMARKS = ROOT / "benchmarks"            # KoMT-Bench, IFEval-Ko 공식 코드와 데이터

V2_DATASET = ROOT / "prompt-optimizer-v2" / "data" / "dataset.jsonl"   # train 923 / val 54 / test 54
KOMT_QUESTIONS = BENCHMARKS / "KoMT-Bench" / "FastChat" / "fastchat" / "llm_judge" / "data" / "mt_bench" / "question_ko.jsonl"

MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]   # 답변 모델 (Ollama)


# ---------- 2. 파일 ----------
def read_jsonl(path):
    """jsonl 파일의 모든 줄을 읽는다. 파일이 없으면 빈 목록."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """파일을 새로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    """파일 끝에 한 줄 덧붙인다. 중간에 끊겨도 다시 실행하면 이어서 할 수 있다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def v2_split(name):
    """v2 데이터셋에서 split이 name(train, val, test)인 행"""
    rows = []
    for row in read_jsonl(V2_DATASET):
        if row["split"] == name:
            rows.append(row)
    return rows


def is_similar_to_any(text, others):
    """others 중 하나라도 text와 60% 이상 비슷한가 (평가 문항이나 기존 질문과 겹치지 않게 할 때 쓴다)"""
    for other in others:
        if difflib.SequenceMatcher(None, text, other).ratio() >= 0.6:
            return True
    return False


def run_in_parallel(function, items):
    """GPT를 부르는 함수를 8개씩 동시에 실행한다. 결과 순서는 items 순서와 같다."""
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(function, items))


# ---------- 3. 통계와 표 ----------
def paired_test(diffs):
    """
    같은 문항끼리의 점수 차이 목록 -> (평균, 95% 신뢰구간 하한, 상한, p값)

    신뢰구간: 차이를 복원 추출해서 평균 내기를 1만 번 반복한다 (부트스트랩).
    p값: 윌콕슨 부호 순위 검정 (양측). 차이가 0인 쌍은 순위 합을 양쪽에 반씩 나눈다 (zsplit).
    """
    diffs = np.array(diffs, dtype=float)
    rng = np.random.default_rng(0) # seed를 0으로 하여 난수 생성
    samples = rng.choice(diffs, size=(10000, len(diffs)))
    sample_means = samples.mean(axis=1)
    low = np.percentile(sample_means, 2.5)
    high = np.percentile(sample_means, 97.5) # 1만개 부트스트랩

    if np.all(diffs == 0):
        p = 1.0   # 차이가 전부 0이면 검정할 것이 없다
    else: # z split 윌콕슨 부호 순위 검정
        p = stats.wilcoxon(diffs, zero_method="zsplit").pvalue
    return diffs.mean(), low, high, p


def format_delta(diffs):
    """'-1.24 [-1.98, -0.52], p 0.001' 형식의 글자"""
    mean, low, high, p = paired_test(diffs)
    return f"{mean:+.2f} [{low:+.2f}, {high:+.2f}], p {p:.3f}"


def win_rate(diffs):
    """이긴 비율. 이기면 1, 비기면 0.5, 지면 0점으로 평균 낸다."""
    points = []
    for d in diffs:
        if d > 0:
            points.append(1.0)
        elif d == 0:
            points.append(0.5)
        else:
            points.append(0.0)
    return np.mean(points)


def table(header, rows):
    """마크다운 표 글자"""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# ---------- 4. 외부 모델 ----------
# 모델 라이브러리(openai, ollama, torch)는 함수 안에서 불러온다.
# 그래야 GPT만 쓰는 단계는 torch가 없어도 돌아간다.

def gpt(prompt, model="gpt-4o-mini-2024-07-18", system=None, temperature=0.0, json_mode=False):
    """GPT에 한 번 묻고 답 글자를 돌려준다. 환경 변수 OPENAI_API_KEY가 필요하다."""
    from openai import OpenAI

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if json_mode:
        response = OpenAI().chat.completions.create(model=model, messages=messages, temperature=temperature, seed=42,
                                                    response_format={"type": "json_object"})
    else:
        response = OpenAI().chat.completions.create(model=model, messages=messages, temperature=temperature, seed=42)
    return response.choices[0].message.content.strip()


def answer(model, messages, temperature=0.0, max_tokens=2048):
    """Ollama 답변 모델로 답을 만든다.

    같은 말이 계속 반복되면 Ollama가 오류를 내고 생성을 멈춘다.
    그때는 멈추기 전까지 나온 글을 답으로 쓴다.
    """
    import ollama

    options = {"temperature": temperature, "seed": 42, "num_ctx": 8192, "num_predict": max_tokens}
    text = ""
    try:
        for chunk in ollama.chat(model=model, messages=messages, stream=True, options=options):
            text += chunk.message.content
    except ollama.ResponseError as error:
        if "repeat" not in str(error):
            raise
    return text


def unload(model):
    """다음 모델을 올리기 전에 Ollama 모델을 GPU에서 내린다."""
    import subprocess
    subprocess.run(["ollama", "stop", model])


class RewardModel:
    """보상 모델 Skywork-Reward-V2-Qwen3-4B. 질문과 답을 넣으면 점수 하나를 낸다 (높을수록 좋은 답)."""

    NAME = "Skywork/Skywork-Reward-V2-Qwen3-4B"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.NAME, dtype=torch.bfloat16, device_map="cuda:0", num_labels=1)
        self.model.eval()

    def score(self, question, answer):
        conversation = [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer}]
        inputs = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt", return_dict=True)
        inputs = inputs.to("cuda:0")
        with self.torch.no_grad():
            output = self.model(**inputs)
        return output.logits[0][0].item()
