"""
v4 평가 (연구일지 5장)

v4 프롬프트로 받은 답이 원래 질문으로 받은 답보다 나은지 본다. 평가 문항은 v3와 같다.

순서
  1. v4로 최적화 프롬프트 만들기
  2. 답변 모델이 원래 질문과 최적화 프롬프트로 각각 답하기
  3. 채점: 보상 모델, WildChat GPT 1:1 판정, KoMT gpt-4o, IFEval-Ko 규칙
  4. 결과표 docs/결과_v4.md

  python prompt-optimizer-v4/eval/evaluate.py prompts            # 1. (GPU)
  python prompt-optimizer-v4/eval/evaluate.py answers            # 2. (Ollama)
  python prompt-optimizer-v4/eval/evaluate.py reward             # 3. 보상 모델 (GPU)
  python prompt-optimizer-v4/eval/evaluate.py wildchat-judge     # 3. WildChat GPT 판정 (API)
  python prompt-optimizer-v4/eval/evaluate.py komt-judge         # 3. KoMT gpt-4o 채점 (API)
  python prompt-optimizer-v4/eval/evaluate.py ifeval             # 3. IFEval-Ko 규칙 채점
  python prompt-optimizer-v4/eval/evaluate.py report             # 4.

평가 세트
  test54   v2 데이터셋 test 54문항
  komt     KoMT-Bench 80문항 (1턴 질문만 최적화하고, 2턴 질문은 원래 그대로)
  wildchat WildChat 짧은 한국어 질문 300문항 (v3 평가 때 고른 문항, results/v3/wildchat_questions/)
  ifeval   IFEval-Ko 342문항

결과: results/v4/{세트}/
  optimized_prompts.jsonl                                      v4가 만든 프롬프트
  original_answers.jsonl, optimized_answers.jsonl              원래 질문으로 받은 답, v4 프롬프트로 받은 답
  original_reward_scores.jsonl, optimized_reward_scores.jsonl  보상 모델 점수
  gpt_judgments.jsonl (komt는 original_gpt_judgments.jsonl도)  GPT 채점
  original_checker_scores.jsonl, checker_scores.jsonl          IFEval-Ko 규칙 채점

결과 파일 한 줄의 cond는 어떤 프롬프트로 받은 답인지를 적은 것이다 ('original' 또는 'v4').
모든 채점은 원래 질문 기준이다. 최적화 프롬프트로 받은 답도 원래 질문에 대한 답으로 채점한다.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
import time
import types
import uuid

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 버전 폴더의 utils.py, optimizer.py를 불러오기 위해

from utils import (BENCHMARKS, DOCS, EVAL, KOMT_QUESTIONS, MODELS, RESULTS, WILDCHAT_QUESTIONS, RewardModel, answer,
                   append_jsonl, format_delta, gpt, read_jsonl, run_in_parallel, table, unload, v2_split, win_rate,
                   write_jsonl)

VERSION = "v4"
WILDCHAT_JUDGE = "gpt-4o-mini-2024-07-18"
IFEVAL_DATA = BENCHMARKS / "ifeval_ko" / "data" / "train-00000-of-00001.parquet"
KOMT_DATA = BENCHMARKS / "KoMT-Bench" / "FastChat" / "fastchat" / "llm_judge" / "data" / "mt_bench"

# 평가 세트마다 쓴 답변 모델.
# KoMT는 gpt-4o 채점 비용 때문에 3B만 했다.
# IFEval은 v4가 342문항을 하나도 바꾸지 않아서(모두 gate에서 걸러짐) 3B만 확인했다.
ANSWER_MODELS = {
    "test54": MODELS,
    "komt": ["qwen2.5:3b"],
    "wildchat": MODELS,
    "ifeval": ["qwen2.5:3b"],
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
    elif name == "wildchat":
        for r in read_jsonl(WILDCHAT_QUESTIONS):
            rows.append({"id": r["id"], "question": r["q"]})
    elif name == "ifeval":
        import pandas as pd
        for r in pd.read_parquet(IFEVAL_DATA).itertuples():
            rows.append({"id": int(r.key), "question": r.prompt})
    return rows


def make_prompts():
    """세트마다 results/v4/{세트}/optimized_prompts.jsonl을 만든다. {version, id, original, optimized}"""
    from optimizer import load_optimizer
    optimize = load_optimizer()   # gate()를 통과한 질문만 모델로 다시 쓴다
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
    cond가 'original'이면 원래 질문으로, 'v4'이면 최적화 프롬프트로 묻는다."""
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
        elif name == "ifeval":
            row["answer"] = answer(model, [{"role": "user", "content": question}], 0.0, 1280)   # IFEval-Ko 공식 설정
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
    for name in ["test54", "komt", "wildchat"]:
        folder = RESULTS / name
        score_with_reward_model(reward, name, folder / "original_answers.jsonl", folder / "original_reward_scores.jsonl")
        score_with_reward_model(reward, name, folder / "optimized_answers.jsonl", folder / "optimized_reward_scores.jsonl")


# ========== 3-2. WildChat GPT 1:1 판정 ==========
# Arena-Hard-Auto 공식 1:1 비교 프롬프트 (그대로)
JUDGE_SYSTEM = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""
JUDGE_USER = """<|User Prompt|>
{q}

<|The Start of Assistant A's Answer|>
{a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{b}
<|The End of Assistant B's Answer|>"""


def answers_by_id(path, model):
    answers = {}
    for r in read_jsonl(path):
        if r["model"] == model:
            answers[r["id"]] = r["answer"]
    return answers


def judge_wildchat(model):
    """원래 질문의 답과 v4 프롬프트의 답 중 어느 쪽이 나은지 gpt-4o-mini가 판정한다.
    순서에 따른 치우침을 없애려고 문항마다 A, B 순서를 바꿔 두 번 판정한다.
      order 'cond_first': A가 최적화 답, B가 원문 답
      order 'orig_first': A가 원문 답, B가 최적화 답"""
    folder = RESULTS / "wildchat"
    original = answers_by_id(folder / "original_answers.jsonl", model)
    optimized = answers_by_id(folder / "optimized_answers.jsonl", model)
    judgments_path = folder / "gpt_judgments.jsonl"

    done = set()
    for r in read_jsonl(judgments_path):
        if r["model"] == model and r["verdict"]:
            done.add((r["id"], r["order"]))

    jobs = []
    for q in load_questions("wildchat"):
        for order in ["cond_first", "orig_first"]:
            if (q["id"], order) not in done:
                jobs.append((q, order))

    def judge_one(job):
        q, order = job
        if order == "cond_first":
            a, b = optimized[q["id"]], original[q["id"]]
        else:
            a, b = original[q["id"]], optimized[q["id"]]
        text = gpt(JUDGE_USER.format(q=q["question"], a=a, b=b), model=WILDCHAT_JUDGE, system=JUDGE_SYSTEM)
        verdicts = re.findall(r"\[\[([AB<>=]+)\]\]", text)   # 예: "[[A>B]]" -> "A>B"
        verdict = verdicts[-1] if verdicts else None
        return {"version": VERSION, "model": model, "id": q["id"], "order": order, "verdict": verdict, "judgment": text}

    for row in run_in_parallel(judge_one, jobs):
        append_jsonl(judgments_path, row)


def judge_all_wildchat():
    for model in ANSWER_MODELS["wildchat"]:
        judge_wildchat(model)


# ========== 3-3. KoMT-Bench gpt-4o 채점 ==========
def judge_komt(model, cond, answers_path, judgments_path):
    """KoMT-Bench 공식 채점 코드(komt_judge.py)로 gpt-4o 채점을 하고, 결과를 judgments_path에 옮긴다."""
    model_id = model.replace(":", "-") + "__" + cond   # 공식 코드에서 쓰는 이름. 예: qwen2.5-3b__v4

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


# ========== 3-4. IFEval-Ko 규칙 채점 ==========
def load_ifeval_checker():
    """IFEval-Ko 공식 채점 코드를 불러온다.

    공식 코드는 lm-evaluation-harness 안에 있다는 전제로 'lm_eval.tasks.ifeval_ko' 경로로 서로를 불러온다.
    harness 전체를 설치하지 않고, 빈 'lm_eval' 패키지를 만들어 benchmarks/ifeval_ko를 그 경로로 연결한다.
    """
    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__path__ = []
    sys.modules["lm_eval"] = lm_eval

    lm_eval_tasks = types.ModuleType("lm_eval.tasks")
    lm_eval_tasks.__path__ = [str(BENCHMARKS / "ifeval_ko")]
    sys.modules["lm_eval.tasks"] = lm_eval_tasks

    from lm_eval.tasks.ifeval_ko import utils
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0   # 공식 코드의 언어 감지는 난수를 써서, 고정하지 않으면 실행마다 결과가 조금 달라진다
    return utils


def to_plain(value):
    """parquet에서 읽은 값을 원래 JSON 값으로 되돌린다 (3.0 -> 3, numpy 배열 -> 목록)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        return list(value)
    return value


def load_ifeval_docs():
    """문항 번호 -> 공식 채점 코드가 받는 문항 정보"""
    import pandas as pd
    docs = {}
    for r in pd.read_parquet(IFEVAL_DATA).itertuples():
        kwargs = []
        for kw in r.kwargs:
            kwargs.append({key: to_plain(value) for key, value in kw.items()})
        docs[int(r.key)] = {"key": int(r.key), "prompt": r.prompt,
                            "instruction_id_list": list(r.instruction_id_list), "kwargs": kwargs}
    return docs


def check_ifeval(checker, docs, answers_path, scores_path):
    """문항마다 지시를 모두 지켰는지 채점한다. {model, cond, id, strict, loose}"""
    rows = []
    for r in read_jsonl(answers_path):
        result = checker.process_results(docs[r["id"]], [r["answer"]])
        rows.append({"model": r["model"], "cond": r["cond"], "id": r["id"],
                     "strict": result["prompt_level_strict_acc"], "loose": result["prompt_level_loose_acc"]})
    write_jsonl(scores_path, rows)


def check_all_ifeval():
    checker = load_ifeval_checker()
    docs = load_ifeval_docs()
    folder = RESULTS / "ifeval"
    check_ifeval(checker, docs, folder / "original_answers.jsonl", folder / "original_checker_scores.jsonl")
    check_ifeval(checker, docs, folder / "optimized_answers.jsonl", folder / "checker_scores.jsonl")


# ========== 4. 결과표 ==========
# 점수 차이: 평균 Δ [95% 신뢰구간], 윌콕슨 p
# GPT 판정: 승률 [95% 신뢰구간], 부호 검정 p
# IFEval: 정답률 차이(%p), McNemar p
# 모두 원래 질문 대비

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


def rewritten_komt_ids():
    """KoMT에서 실제로 강화한 문항. 원문을 그대로 둔 문항은 뺀다."""
    ids = set()
    for r in read_jsonl(RESULTS / "komt" / "optimized_prompts.jsonl"):
        if r["optimized"].strip() == r["original"].strip():
            continue
        ids.add(r["id"])
    return ids


VERDICT_SCORE = {"A>>B": 1.0, "A>B": 1.0, "A=B": 0.5, "B>A": 0.0, "B>>A": 0.0}   # A가 받는 점수. 크게 이겨도 1승


def wildchat_cell(model):
    """WildChat GPT 판정 승률. 문항 점수 = 두 순서 판정의 평균 (최적화 쪽이 이기면 1, 비기면 0.5, 지면 0)"""
    # a_points[(id, order)] = 그 판정에서 A가 받은 점수
    a_points = {}
    for r in read_jsonl(RESULTS / "wildchat" / "gpt_judgments.jsonl"):
        if r["model"] == model and r["verdict"] in VERDICT_SCORE:
            a_points[(r["id"], r["order"])] = VERDICT_SCORE[r["verdict"]]

    ids = set()
    for (question_id, order) in a_points:
        ids.add(question_id)

    question_scores = []
    for question_id in sorted(ids):
        if (question_id, "cond_first") not in a_points or (question_id, "orig_first") not in a_points:
            continue   # 두 순서 판정이 모두 있어야 쓴다
        optimized_as_a = a_points[(question_id, "cond_first")]       # cond_first에서 최적화 쪽은 A
        optimized_as_b = 1 - a_points[(question_id, "orig_first")]   # orig_first에서 최적화 쪽은 B
        question_scores.append((optimized_as_a + optimized_as_b) / 2)

    if not question_scores:
        return "-"

    s = np.array(question_scores)
    sample_means = np.random.default_rng(0).choice(s, size=(10000, len(s))).mean(axis=1)   # 부트스트랩
    low = np.percentile(sample_means, 2.5)
    high = np.percentile(sample_means, 97.5)
    wins = int((s > 0.5).sum())
    losses = int((s < 0.5).sum())
    p = binomtest(wins, wins + losses).pvalue   # 부호 검정 (비긴 문항은 뺀다)
    cell = f"{s.mean():.1%} [{low:.1%}, {high:.1%}], p {p:.3f}"
    if len(s) != 300:
        cell += f" (n={len(s)})"
    return cell


def report():
    rewritten = rewritten_komt_ids()
    rows = [
        ["WildChat 300 GPT 판정 승률"] + [wildchat_cell(m) for m in MODELS],
        ["WildChat 300 보상 모델 Δ"] + [reward_cell("wildchat", m) for m in MODELS],
        ["test 54 보상 모델 Δ"] + [reward_cell("test54", m) for m in MODELS],
        ["KoMT 1턴 gpt-4o 점수 Δ"] + [komt_cell(m) for m in MODELS],
        [f"KoMT 1턴 gpt-4o 점수 Δ (강화한 {len(rewritten)}문항만)"] + [komt_cell(m, ids=rewritten) for m in MODELS],
    ]
    # IFEval은 바뀐 문항이 0개라 원문과 같아서 표에 넣지 않는다
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
    elif command == "wildchat-judge":
        judge_all_wildchat()
    elif command == "komt-judge":
        judge_all_komt()
    elif command == "ifeval":
        check_all_ifeval()
    elif command == "report":
        report()
    else:
        print(__doc__)
