"""v3 설계 실험 (연구일지 2장). v2 평가에서 손해가 난 원인을 찾고, v3 데이터 형식을 정하려고 했다.
문항은 v2 val 54문항 (test는 평가용이라 쓰지 않았다).

  python prompt-optimizer-v3/eval/experiments.py prompts   # 1. v2 모델 출력과 섹션 제거 변형 만들기 (GPU)
  python prompt-optimizer-v3/eval/experiments.py gpt       # 2. gpt-4o로 C, D 프롬프트 만들기 (API)
  python prompt-optimizer-v3/eval/experiments.py answer    # 3. 답변 모델로 답하기 (Ollama)
  python prompt-optimizer-v3/eval/experiments.py score     # 4. 보상 모델 채점 (GPU)
  python prompt-optimizer-v3/eval/experiments.py report    # 5. docs/결과_설계실험.md

섹션 제거 실험: v2 출력에서 섹션을 하나씩 빼거나 더한 변형. 답변 모델 7B만.
구조화 실험: 원문, v2 데이터셋 정답, v2 모델 출력, C, D. 답변 모델 3B, 7B, 14B.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 버전 폴더의 utils.py, optimizer.py를 불러오기 위해

from utils import (DOCS, MODELS, RESULTS, RewardModel, answer, append_jsonl, format_delta, gpt, read_jsonl, table,
                   unload, v2_split, win_rate)

PROMPTS = RESULTS / "experiments" / "variant_prompts.jsonl"   # {id, variant, prompt}
ANSWERS = RESULTS / "experiments" / "answers.jsonl"           # {id, variant, model, answer}
SCORES = RESULTS / "experiments" / "reward_scores.jsonl"      # {id, variant, model, score}

ABLATION_MODEL = "qwen2.5:7b"   # 섹션 제거 실험의 답변 모델

# 변형 이름: 결과표에 쓸 설명
STRUCTURE_VARIANTS = {
    "gold": "v2 데이터셋 정답",
    "v2": "v2 모델 출력",
    "gpt_grounded": "C 근거 있는 구조화",
    "gpt_minimal": "D 원문 + 추가 조건",
}
ABLATION_VARIANTS = {
    "v2-역할": "[역할] 삭제",
    "v2-수행단계": "[수행 단계] 삭제",
    "v2-출력형식": "[출력 형식] 삭제",
    "v2-분량줄": "분량 줄 삭제",
    "v2-빈칸줄": "{{빈칸}} 줄 삭제",
    "v2+원래요청": "[원래 요청] 추가",
}


# ---------- 1. v2 출력과 섹션 제거 ----------
def split_sections(prompt):
    """프롬프트를 '[이름]'으로 시작하는 줄마다 끊는다. [(이름, 섹션 글), ...]
    첫 '[이름]' 앞에 글이 있으면 이름이 ''인 섹션이 된다."""
    sections = []
    for line in prompt.strip().splitlines():
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            sections.append({"name": header.group(1), "lines": [line]})
        elif not sections:
            sections.append({"name": "", "lines": [line]})
        else:
            sections[-1]["lines"].append(line)

    result = []
    for section in sections:
        result.append((section["name"], "\n".join(section["lines"]).strip()))
    return result


def drop_section(prompt, name):
    """이름이 name인 섹션을 지운다."""
    kept = []
    for section_name, text in split_sections(prompt):
        if section_name != name:
            kept.append(text)
    return "\n\n".join(kept)


def drop_lines(prompt, pattern):
    """pattern이 들어 있는 줄을 지운다."""
    kept = []
    for line in prompt.splitlines():
        if not re.search(pattern, line):
            kept.append(line)
    return "\n".join(kept)


def ablations(v2, original):
    """v2 출력 하나로 섹션 제거 변형들을 만든다."""
    return {
        "v2": v2,
        "v2-역할": drop_section(v2, "역할"),
        "v2-수행단계": drop_section(v2, "수행 단계"),
        "v2-출력형식": drop_section(v2, "출력 형식"),
        "v2-분량줄": drop_lines(v2, r"^\s*-\s*분량\s*:"),
        "v2-빈칸줄": drop_lines(v2, r"\{\{.*?\}\}"),
        "v2+원래요청": v2.rstrip() + "\n\n[원래 요청]\n" + original.strip(),
    }


def make_prompts():
    from optimizer import V2_ADAPTER, PromptOptimizer
    optimize = PromptOptimizer(V2_ADAPTER)   # v2 모델 출력
    for row in v2_split("val"):
        append_jsonl(PROMPTS, {"id": row["id"], "variant": "original", "prompt": row["input"]})
        append_jsonl(PROMPTS, {"id": row["id"], "variant": "gold", "prompt": row["output"]})
        v2_output = optimize(row["input"])
        for variant, prompt in ablations(v2_output, row["input"]).items():
            append_jsonl(PROMPTS, {"id": row["id"], "variant": variant, "prompt": prompt})


# ---------- 2. gpt-4o 프롬프트 (C, D) ----------
GROUNDED = """너는 프롬프트 개선기다. 사용자의 원래 요청에 답하지 말고, 다른 언어모델이 이 요청에 더 좋은 답을 하도록 요청을 다시 써라.
규칙:
1. 원래 요청의 사실·조건·숫자·의도를 바꾸거나 빼지 않는다.
2. 원래 요청에 없는 분량·톤 제약, 새로운 사실, 특정 풀이 방법을 만들지 않는다. 과제 성격상 명백한 것만 허용한다 (예: 비교 요청이면 기준별로 정리).
3. 역할(페르소나, "당신은 ~입니다")을 쓰지 않는다.
4. {{빈칸}}을 쓰지 않는다. 정보가 부족하면 합리적인 기본값을 "가정"으로 적는다.
5. 아래 섹션 중 이 요청에 실제로 도움이 되는 것만 쓴다: [과제], [맥락 및 가정], [수행 단계], [출력 형식]
6. 맨 끝에 [원래 요청] 섹션을 두고 원래 요청을 글자 그대로 옮긴다.
개선된 프롬프트만 출력하라."""

MINIMAL = """너는 프롬프트 개선기다. 사용자의 원래 요청에 답하지 말라.
원래 요청은 그대로 두고, 다른 언어모델이 더 좋은 답을 하도록 요청 끝에 덧붙일 "추가 조건"을 0~3줄 써라.
각 줄은 다음 세 조건을 모두 만족해야 한다.
1. 근거가 있다: 원래 요청에서 도출되거나 누구나 동의할 기본값이다.
2. 답을 바꾼다: 이 조건이 없으면 답이 눈에 띄게 달라진다.
3. 사용자가 거부할 이유가 없다: 요청하지 않은 범위 축소·분량 제한·톤 강요가 아니다.
덧붙일 것이 없으면 "없음"만 출력하라. 그 외에는 "- "로 시작하는 줄만 출력하라."""


def make_gpt_prompts():
    for row in v2_split("val"):
        original = row["input"].strip()
        user = f"<원래 요청>\n{original}\n</원래 요청>"

        # C: 근거 있는 구조화
        grounded = gpt(user, model="gpt-4o-2024-08-06", system=GROUNDED)

        # D: 원문 끝에 "추가 조건" 줄만 붙인다
        reply = gpt(user, model="gpt-4o-2024-08-06", system=MINIMAL)
        extra_lines = []
        for line in reply.splitlines():
            if line.strip().startswith("-"):
                extra_lines.append(line)
        minimal = original
        if extra_lines:
            minimal += "\n\n추가 조건:\n" + "\n".join(extra_lines)

        append_jsonl(PROMPTS, {"id": row["id"], "variant": "gpt_grounded", "prompt": grounded})
        append_jsonl(PROMPTS, {"id": row["id"], "variant": "gpt_minimal", "prompt": minimal})


# ---------- 3. 답변 ----------
def answer_models(variant):
    """구조화 실험(원문 포함)은 세 모델, 섹션 제거 실험은 7B만"""
    if variant == "original" or variant in STRUCTURE_VARIANTS:
        return MODELS
    return [ABLATION_MODEL]


def make_answers():
    done = set()
    for r in read_jsonl(ANSWERS):
        done.add((r["id"], r["variant"], r["model"]))

    for model in MODELS:
        for p in read_jsonl(PROMPTS):
            if model not in answer_models(p["variant"]):
                continue
            if (p["id"], p["variant"], model) in done:
                continue
            text = answer(model, [{"role": "user", "content": p["prompt"]}])
            append_jsonl(ANSWERS, {"id": p["id"], "variant": p["variant"], "model": model, "answer": text})
        unload(model)


# ---------- 4. 채점 ----------
def score_answers():
    # 채점은 항상 원래 질문 기준이다
    question = {}
    for row in v2_split("val"):
        question[row["id"]] = row["input"]

    done = set()
    for r in read_jsonl(SCORES):
        done.add((r["id"], r["variant"], r["model"]))

    reward = RewardModel()
    for r in read_jsonl(ANSWERS):
        if (r["id"], r["variant"], r["model"]) in done:
            continue
        score = reward.score(question[r["id"]], r["answer"])
        append_jsonl(SCORES, {"id": r["id"], "variant": r["variant"], "model": r["model"], "score": score})


# ---------- 5. 결과표 ----------
def score_differences(scores, model, variant, base):
    """같은 문항에서 (variant 점수 - base 점수) 목록. scores[(id, variant, model)] = 점수"""
    base_ids = set()
    for (question_id, v, m) in scores:
        if v == base and m == model:
            base_ids.add(question_id)

    differences = []
    for question_id in sorted(base_ids):
        if (question_id, variant, model) in scores:
            differences.append(scores[(question_id, variant, model)] - scores[(question_id, base, model)])
    return differences


def report():
    scores = {}
    for r in read_jsonl(SCORES):
        scores[(r["id"], r["variant"], r["model"])] = r["score"]

    lines = ["# v3 설계 실험 (v2 val 54문항)\n",
             "채점: Skywork-Reward-V2-Qwen3-4B, (원래 질문, 답변). 값은 평균 Δ [95% 신뢰구간], 윌콕슨 p.\n"]

    # 섹션 제거 실험
    lines.append("## 섹션 제거 실험 (답변 모델 7B)\n")
    rows = [["v2 그대로", "-", format_delta(score_differences(scores, ABLATION_MODEL, "v2", "original"))]]
    for variant, label in ABLATION_VARIANTS.items():
        vs_v2 = format_delta(score_differences(scores, ABLATION_MODEL, variant, "v2"))
        vs_original = format_delta(score_differences(scores, ABLATION_MODEL, variant, "original"))
        rows.append([label, vs_v2, vs_original])
    lines.append(table(["조건", "v2 대비 Δ", "원문 대비 Δ"], rows) + "\n")

    # 구조화 실험: 점수 차이
    lines.append("## 구조화 실험 (원문 대비 Δ)\n")
    rows = []
    for variant, label in STRUCTURE_VARIANTS.items():
        row = [label]
        for model in MODELS:
            row.append(format_delta(score_differences(scores, model, variant, "original")))
        rows.append(row)
    lines.append(table(["조건", "3B", "7B", "14B"], rows) + "\n")

    # 구조화 실험: 승률
    lines.append("## 구조화 실험 원문 대비 승률\n")
    rows = []
    for variant, label in STRUCTURE_VARIANTS.items():
        row = [label]
        for model in MODELS:
            row.append(f"{win_rate(score_differences(scores, model, variant, 'original')):.0%}")
        rows.append(row)
    lines.append(table(["조건", "3B", "7B", "14B"], rows) + "\n")

    text = "\n".join(lines)
    (DOCS / "결과_설계실험.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prompts":
        make_prompts()
    elif command == "gpt":
        make_gpt_prompts()
    elif command == "answer":
        make_answers()
    elif command == "score":
        score_answers()
    elif command == "report":
        report()
    else:
        print(__doc__)
