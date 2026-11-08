"""
v2 평가 (연구일지 1장)

v2 프롬프트로 받은 답이 원래 질문으로 받은 답보다 나은지 본다.

순서
  1. v2로 최적화 프롬프트 만들기 (inference.py)
  2. 답변 모델 3B, 7B, 14B가 원래 질문과 최적화 프롬프트로 각각 답하기
  3. 채점: 보상 모델(test54, komt), KoMT gpt-4o
  4. 결과표 docs/결과_v2.md

  python prompt-optimizer-v2/eval/evaluate.py prompts      # 1. (GPU)
  python prompt-optimizer-v2/eval/evaluate.py answers      # 2. (Ollama)
  python prompt-optimizer-v2/eval/evaluate.py reward       # 3. 보상 모델 (GPU)
  python prompt-optimizer-v2/eval/evaluate.py komt-judge   # 3. KoMT gpt-4o 채점 (API)
  python prompt-optimizer-v2/eval/evaluate.py report       # 4.

평가 세트
  test54   v2 데이터셋 test 54문항
  komt     KoMT-Bench 80문항 (1턴 질문만 최적화하고, 2턴 질문은 원래 그대로)

결과: results/v2/{세트}/
  optimized_prompts.jsonl                                      v2가 만든 프롬프트
  original_answers.jsonl, optimized_answers.jsonl              원래 질문으로 받은 답, v2 프롬프트로 받은 답
  original_reward_scores.jsonl, optimized_reward_scores.jsonl  보상 모델 점수
  original_gpt_judgments.jsonl, gpt_judgments.jsonl            KoMT gpt-4o 채점

결과 파일 한 줄의 cond는 어떤 프롬프트로 받은 답인지를 적은 것이다 ('original' 또는 'v2').
모든 채점은 원래 질문 기준이다. 최적화 프롬프트로 받은 답도 원래 질문에 대한 답으로 채점한다.
"""
import os
import subprocess
import sys
from pathlib import Path
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 버전 폴더의 utils.py, optimizer.py를 불러오기 위해

from utils import (BENCHMARKS, DOCS, EVAL, KOMT_QUESTIONS, MODELS, RESULTS, RewardModel, answer, append_jsonl,
                   format_delta, read_jsonl, table, unload, v2_split, win_rate, write_jsonl)

VERSION = "v2"
KOMT_DATA = BENCHMARKS / "KoMT-Bench" / "FastChat" / "fastchat" / "llm_judge" / "data" / "mt_bench"

# 평가 세트마다 쓴 답변 모델
ANSWER_MODELS = {
    "test54": MODELS,
    "komt": MODELS,
}


# ========== 1. 최적화 프롬프트 ==========
def load_questions(name):
    """평가 세트의 문항 목록 [{id, question}]"""
    rows = []
    if name == "test54":
        for r in v2_split("test"):
            rows.append({"id": r["id"], "question": r["input"]})
    elif name == "komt":
        for q in read_jsonl(KOMT_QUESTIONS):
            rows.append({"id": q["question_id"], "question": q["turns"][0]})
    return rows


def make_prompts():
    """세트마다 results/v2/{세트}/optimized_prompts.jsonl을 만든다. {version, id, original, optimized}"""
    from inference import PromptOptimizer
    optimize = PromptOptimizer()
    for name in ANSWER_MODELS:
        path = RESULTS / name / "optimized_prompts.jsonl"
        done = set()
        for r in read_jsonl(path):
            done.add(r["id"])
        for q in load_questions(name):
            if q["id"] in done:
                continue
            row = {"version": VERSION, "id": q["id"], "original": q["question"], "optimized": optimize(q["question"])}
            append_jsonl(path, row)


# ========== 2. 답변 ==========
# KoMT-Bench(FastChat) 공식 카테고리별 temperature
KOMT_TEMPERATURE = {"writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0,
                    "coding": 0.0, "reasoning": 0.0, "stem": 0.1, "humanities": 0.1}


def answer_komt(model, first_question, komt_question):
    """KoMT는 2턴 대화다. 1턴은 주어진 질문, 2턴은 원래 2턴 질문 그대로."""
    temperature = KOMT_TEMPERATURE[komt_question["category"]]
    messages = [{"role": "user", "content": first_question}]
    first_answer = answer(model, messages, temperature)
    messages.append({"role": "assistant", "content": first_answer})
    messages.append({"role": "user", "content": komt_question["turns"][1]})
    second_answer = answer(model, messages, temperature)
    return first_answer, second_answer


def make_answers(model, name, cond, path):
    """답변 모델로 답해서 path에 쓴다. {model, cond, id, input, answer}
    cond가 'original'이면 원래 질문으로, 'v2'이면 최적화 프롬프트로 묻는다."""
    komt_questions = {}
    for q in read_jsonl(KOMT_QUESTIONS):
        komt_questions[q["question_id"]] = q

    done = set()
    for r in read_jsonl(path):
        if r["model"] == model:
            done.add(r["id"])

    for p in read_jsonl(RESULTS / name / "optimized_prompts.jsonl"):
        if p["id"] in done:
            continue
        if cond == "original":
            question = p["original"]
        else:
            question = p["optimized"]

        row = {"model": model, "cond": cond, "id": p["id"], "input": question}
        if name == "komt":
            row["answer"], row["turn2_answer"] = answer_komt(model, question, komt_questions[p["id"]])
        else:
            row["answer"] = answer(model, [{"role": "user", "content": question}], 0.0, 2048)
        append_jsonl(path, row)


def make_all_answers():
    for model in MODELS:
        for name, models in ANSWER_MODELS.items():
            if model in models:
                make_answers(model, name, "original", RESULTS / name / "original_answers.jsonl")
                make_answers(model, name, VERSION, RESULTS / name / "optimized_answers.jsonl")
        unload(model)


# ========== 3-1. 보상 모델 채점 ==========
def score_with_reward_model(reward, name, answers_path, scores_path):
    """답을 (원래 질문, 답)으로 채점해서 scores_path에 쓴다. KoMT는 1턴 답만."""
    question = {}
    for q in load_questions(name):
        question[q["id"]] = q["question"]

    done = set()
    for r in read_jsonl(scores_path):
        done.add((r["model"], r["id"]))

    for r in read_jsonl(answers_path):
        if (r["model"], r["id"]) in done:
            continue
        score = reward.score(question[r["id"]], r["answer"])
        row = {"model": r["model"], "cond": r["cond"], "id": r["id"], "score": score, "answer_chars": len(r["answer"])}
        append_jsonl(scores_path, row)


def score_rewards():
    reward = RewardModel()
    for name in ["test54", "komt"]:
        folder = RESULTS / name
        score_with_reward_model(reward, name, folder / "original_answers.jsonl", folder / "original_reward_scores.jsonl")
        score_with_reward_model(reward, name, folder / "optimized_answers.jsonl", folder / "optimized_reward_scores.jsonl")


# ========== 3-2. KoMT-Bench gpt-4o 채점 ==========
def judge_komt(model, cond, answers_path, judgments_path):
    """KoMT-Bench 공식 채점 코드(komt_judge.py)로 gpt-4o 채점을 하고, 결과를 judgments_path에 옮긴다."""
    model_id = model.replace(":", "-") + "__" + cond   # 공식 코드에서 쓰는 이름. 예: qwen2.5-3b__v2

    # 1. 공식 코드가 읽는 FastChat 형식으로 답을 넣어 둔다
    fastchat_rows = []
    for r in read_jsonl(answers_path):
        if r["model"] == model:
            fastchat_rows.append({"question_id": r["id"], "answer_id": uuid.uuid4().hex, "model_id": model_id,
                                  "tstamp": time.time(),
                                  "choices": [{"index": 0, "turns": [r["answer"], r["turn2_answer"]]}]})
    write_jsonl(KOMT_DATA / "model_answer" / f"{model_id}.jsonl", fastchat_rows)

    # 2. 공식 채점, 한국어가 아닌 답 0점 처리
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    script = str(EVAL / "komt_judge.py")
    subprocess.run([sys.executable, script, "judge", model_id], env=env, check=True)
    subprocess.run([sys.executable, script, "penalty", model_id], env=env, check=True)

    # 3. 결과를 옮긴다
    for r in read_jsonl(KOMT_DATA / "model_judgment" / f"{model_id}_single_final.jsonl"):
        row = {"model": model, "cond": cond, "id": r["question_id"], "turn": r["turn"],
               "score": r["score"], "judgment": r["judgment"]}
        append_jsonl(judgments_path, row)


def models_in(path):
    """결과 파일에 들어 있는 답변 모델들"""
    models = set()
    for r in read_jsonl(path):
        models.add(r["model"])
    return models


def judge_all_komt():
    folder = RESULTS / "komt"
    for model in ANSWER_MODELS["komt"]:
        if model not in models_in(folder / "original_gpt_judgments.jsonl"):
            judge_komt(model, "original", folder / "original_answers.jsonl", folder / "original_gpt_judgments.jsonl")
        if model not in models_in(folder / "gpt_judgments.jsonl"):
            judge_komt(model, VERSION, folder / "optimized_answers.jsonl", folder / "gpt_judgments.jsonl")


# ========== 4. 결과표 ==========
# 점수 차이: 평균 Δ [95% 신뢰구간], 윌콕슨 p. 원래 질문 대비

def differences(original_rows, optimized_rows, model, value="score", ids=None):
    """같은 문항에서 (최적화 쪽 값 - 원문 쪽 값) 목록. ids를 주면 그 문항만."""
    original = {}
    for r in original_rows:
        if r["model"] == model:
            original[r["id"]] = r[value]
    optimized = {}
    for r in optimized_rows:
        if r["model"] == model:
            optimized[r["id"]] = r[value]

    result = []
    for question_id in optimized:
        if question_id not in original:
            continue
        if ids is not None and question_id not in ids:
            continue
        result.append(optimized[question_id] - original[question_id])
    return result


def reward_cell(name, model):
    """보상 모델 점수 차이"""
    folder = RESULTS / name
    d = differences(read_jsonl(folder / "original_reward_scores.jsonl"),
                    read_jsonl(folder / "optimized_reward_scores.jsonl"), model)
    if not d:
        return "-"
    return format_delta(d) + f" (승률 {win_rate(d):.0%})"


def first_turn_scores(path):
    """KoMT 1턴 채점만. 채점 실패(-1)는 뺀다."""
    rows = []
    for r in read_jsonl(path):
        if r["turn"] == 1 and r["score"] != -1:
            rows.append(r)
    return rows


def komt_cell(model, ids=None):
    """KoMT gpt-4o 점수 차이"""
    folder = RESULTS / "komt"
    d = differences(first_turn_scores(folder / "original_gpt_judgments.jsonl"),
                    first_turn_scores(folder / "gpt_judgments.jsonl"), model, ids=ids)
    if not d:
        return "-"
    return format_delta(d)


def report():
    rows = [
        ["test 54 보상 모델 Δ"] + [reward_cell("test54", m) for m in MODELS],
        ["KoMT 1턴 보상 모델 Δ"] + [reward_cell("komt", m) for m in MODELS],
        ["KoMT 1턴 gpt-4o 점수 Δ"] + [komt_cell(m) for m in MODELS],
    ]
    text = f"# {VERSION} 평가 결과 (원문 대비)\n\n" + table(["평가", "3B", "7B", "14B"], rows) + "\n"
    (DOCS / f"결과_{VERSION}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prompts":
        make_prompts()
    elif command == "answers":
        make_all_answers()
    elif command == "reward":
        score_rewards()
    elif command == "komt-judge":
        judge_all_komt()
    elif command == "report":
        report()
    else:
        print(__doc__)
