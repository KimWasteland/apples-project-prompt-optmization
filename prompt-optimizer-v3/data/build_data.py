"""v3 학습 데이터 만들기.

  python prompt-optimizer-v3/data/build_data.py inputs     # 1. gpt-4o로 새 질문 만들기 (넘기는 예시용 100, 이미 자세한 질문 110)
  python prompt-optimizer-v3/data/build_data.py outputs    # 2. gpt-4o-mini로 질문마다 정답 프롬프트 쓰기 + 규칙 검사
  python prompt-optimizer-v3/data/build_data.py assemble   # 3. 합쳐서 data/build/dataset_raw.jsonl (거르기 전)
  python prompt-optimizer-v3/data/build_data.py answer     # 4. Qwen2.5-7B가 원문, 정답 후보 2개로 각각 답하기
  python prompt-optimizer-v3/data/build_data.py score      # 5. 보상 모델이 (원래 질문, 답)을 채점
  python prompt-optimizer-v3/data/build_data.py filter     # 6. 거른 최종 데이터 dataset_v3.jsonl

정답 형태: "[과제 유형] {과제}" + GPT가 쓴 본문 + "[원래 요청]\n{원문}"
머리말과 [원래 요청]은 원문이 글자 그대로 남도록 코드가 붙인다.

거르기(4~6)
  train만 거르고 val, test는 그대로 둔다.
  후보는 두 개: GPT가 쓴 정답 그대로(v3), 거기서 [출력 형식] 섹션만 지운 것(v3nf).
  두 후보 중 점수가 높은 쪽이 원문 답보다도 높으면 정답으로 쓰고, 둘 다 원문보다 낮으면 학습에서 뺀다.
"""
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 버전 폴더의 utils.py, optimizer.py를 불러오기 위해

from utils import (DATA, KOMT_QUESTIONS, V2_DATASET, RewardModel, answer, append_jsonl, gpt, is_similar_to_any,
                   read_jsonl, run_in_parallel, unload, write_jsonl)

NEW_INPUTS = DATA / "build" / "new_inputs.jsonl"
OUTPUTS = DATA / "build" / "outputs.jsonl"
RAW = DATA / "build" / "dataset_raw.jsonl"
FILTER_ANSWERS = DATA / "build" / "filter_answers.jsonl"
FILTER_SCORES = DATA / "build" / "filter_scores.jsonl"
DATASET = DATA / "dataset_v3.jsonl"


# ---------- 1. 새 질문 ----------
# 고치면 안 되는 질문 (넘기는 예시용)
PASS_GEN = """한국어 사용자가 AI 어시스턴트에게 보낼 법한 요청을 {n}개 만들어라. 종류: {kind}
- 말투를 섞는다: 반말, 존댓말, 구어체, 영어 혼용. 길이도 짧은 것부터 조건이 자세한 것까지 섞는다.
- 실제 사용자처럼 쓴다. 서로 겹치지 않게 다양한 주제로.
JSON으로만: {{"requests": ["...", ...]}}"""
PASS_KINDS = ["코딩 (구현, 디버깅, 코드 설명, SQL)", "수학 계산·풀이 문제",
              "주어진 글이나 데이터에서 정보 추출·변환 (글이나 데이터를 요청 안에 포함)", "논리 퍼즐·추론 문제"]

# 이미 자세한 질문
L3_GEN = """한국어 사용자가 AI 어시스턴트에게 보낼 '이미 구체적인' {task} 요청을 {n}개 만들어라.
- 대상, 목적, 조건, 원하는 형식 중 여러 개가 이미 들어 있는 2~4문장 요청이다.
- 도메인을 다양하게: IT·개발, 교육·학습, 비즈니스·마케팅, 금융·경제, 건강·의료, 여행, 요리·생활, 학술·연구, 법·행정, 문화·엔터, 커리어, 육아·관계
- 말투를 섞는다: 반말, 존댓말, 구어체, 영어 혼용.
JSON으로만: {{"requests": ["...", ...]}}"""
L3_TASKS = ["요약", "설명", "기획", "비교", "추천"]


def make_inputs():
    # 평가 문항(KoMT)이나 기존 질문과 비슷한 것은 뺀다
    existing = []
    for q in read_jsonl(KOMT_QUESTIONS):
        existing.append(q["turns"][0])
    for r in read_jsonl(V2_DATASET):
        existing.append(r["input"])

    # (source, GPT에 보낼 글, task, level, id 앞글자)
    jobs = []
    for kind in PASS_KINDS:
        jobs.append(("pass", PASS_GEN.format(n=25, kind=kind), "기타", "L1", "pt"))
    for task in L3_TASKS:
        jobs.append(("l3", L3_GEN.format(n=22, task=task), task, "L3", "l3x"))

    rows = []
    count = {"pass": 0, "l3": 0}
    for source, prompt, task, level, prefix in jobs:
        reply = gpt(prompt, model="gpt-4o-2024-08-06", system="너는 데이터 생성기다.", temperature=0.9, json_mode=True)
        for text in json.loads(reply)["requests"]:
            text = text.strip()
            if not text or is_similar_to_any(text, existing):
                continue
            existing.append(text)
            count[source] += 1
            rows.append({"id": f"{prefix}_{count[source]:04d}", "source": source, "task": task, "level": level,
                         "input": text, "split": "train"})
    write_jsonl(NEW_INPUTS, rows)


# ---------- 2. 정답 프롬프트 ----------
WRITER = """너는 프롬프트 개선기다. <원래 요청>에 답하지 말고, 다른 언어모델이 이 요청에 더 좋은 답을 하도록 개선된 프롬프트의 본문을 써라.
섹션 (이 순서. [과제]는 필수, 나머지는 이 요청에 실제로 도움이 될 때만 쓰고 아니면 생략한다):
[과제] 요청의 핵심 지시를 1~2문장으로 명확하게.
[맥락 및 가정] 원래 요청에 있는 정보(대상, 조건, 숫자)를 빠짐없이 옮기고, 부족한 부분은 답을 충실하게 만들 합리적인 기본값을 구체적으로 적는다 (예: 일반적인 기간, 포함될 만한 항목).
[수행 단계] 답을 만드는 순서를 2~5단계 번호 목록으로. 이 과제에 맞는 구체적인 단계만.
[출력 형식] 답의 구조만 목록으로 적는다 (예: 개요 / 세부 계획 / 표).
규칙:
1. 원래 요청의 사실·조건·숫자·의도를 바꾸거나 빼지 않는다.
2. 원래 요청에 없는 분량(글자 수·줄 수·개수 제한), 톤·말투 지정, 특정 풀이 방법을 만들지 않는다. 원래 요청이 분량이나 톤을 정했으면 그대로 옮긴다.
3. 역할(페르소나, "당신은 ~입니다")을 쓰지 않는다.
4. {{빈칸}}을 쓰지 않는다.
5. 요청에 포함된 긴 글(요약·검토 대상)은 어느 섹션에도 다시 옮기거나 풀어 쓰지 말고 "[원래 요청]의 글"이라고 가리킨다. 원래 요청은 프롬프트 끝에 그대로 붙는다.
6. 이미 구체적인 요청이면 최소한으로만 정리한다.
7. 원래 요청과 같은 언어로, 정중한 요청문(~해 주세요, ~합니다)으로 쓴다.
섹션 본문만 출력하라. [과제 유형], [역할], [원래 요청]은 쓰지 마라."""

# 분량 표현. 예: "300자", "3~5줄"
QUANTITY = re.compile(r"(\d+)\s*(?:~\s*\d+\s*)?(자|글자|줄|문장|단어|개|단락|문단|쪽|페이지|장|분|초)")
TONE_WORDS = ["톤", "말투", "존댓말", "반말", "격식", "친근", "정중", "문체", "어조"]

# GPT 시도 순서: (모델, temperature). 규칙을 어기면 다음 시도
ATTEMPTS = [("gpt-4o-mini-2024-07-18", 0.0), ("gpt-4o-mini-2024-07-18", 0.7), ("gpt-4o-mini-2024-07-18", 0.7),
            ("gpt-4o-2024-08-06", 0.0)]


def section(body, name):
    """[name] 섹션의 글 (다음 섹션 전까지). 없으면 ''."""
    match = re.search(r"\[" + re.escape(name) + r"\](.*?)(?=\n\s*\[[^\]]+\]|\Z)", body, re.S)
    if match:
        return match.group(1)
    return ""


def check(body, original, task):
    """GPT가 쓴 본문의 규칙 위반 목록. 빈 목록이면 통과."""
    problems = []

    if "[과제]" not in body:
        problems.append("[과제] 섹션 없음")
    for name in ["[과제 유형]", "[역할]", "[원래 요청]", "[원문]"]:
        if name in body:
            problems.append(f"{name} 섹션을 쓰지 말 것")
    if "{{" in body:
        problems.append("{{빈칸}} 사용")
    if re.search(r"당신은|너는 .*(전문가|입니다)", body):
        problems.append("역할(페르소나) 사용")
    if "원래 요청>" in body:
        problems.append("<원래 요청> 태그를 따라 씀")

    # 요약할 긴 글을 본문에 다시 옮겨 썼는가 (40자 넘게 똑같은 부분이 있는가)
    if task == "요약" and len(original) > 150:
        match = difflib.SequenceMatcher(None, body, original, autojunk=False).find_longest_match()
        if match.size >= 40:
            problems.append("요청 속 글을 다시 옮겨 씀 ('[원래 요청]의 글'로 가리킬 것)")

    # 원래 요청에 없는 숫자 분량을 만들었는가 ([수행 단계]의 번호는 빼고 본다)
    body_without_steps = body.replace(section(body, "수행 단계"), "")
    for number, unit in QUANTITY.findall(body_without_steps):
        if number not in original:
            problems.append(f"원래 요청에 없는 분량 '{number}{unit}'")

    # [출력 형식]에 원래 요청에 없는 분량이나 톤을 넣었는가
    output_format = section(body, "출력 형식")
    if "분량" in output_format and not QUANTITY.search(original):
        problems.append("원래 요청에 없는 분량 지정")
    original_mentions_tone = any(word in original for word in TONE_WORDS)
    if re.search(r"톤|말투|어조|문체", output_format) and not original_mentions_tone:
        problems.append("원래 요청에 없는 톤·말투 지정")

    if len(body) > 800:
        problems.append(f"본문이 너무 김 ({len(body)}자 > 800자)")
    return problems


def write_answer(row):
    """정답 본문을 쓰고 검사한다. 규칙을 어기면 어긴 내용을 알려 주고 다시 쓰게 한다.
    gpt-4o-mini 3번이 다 실패하면 gpt-4o로 한 번 더 쓴다. 그래도 실패하면 output은 None."""
    original = row["input"].strip()
    prompt = f"과제 유형: {row['task']}\n<원래 요청>\n{original}\n</원래 요청>"
    feedback = ""
    for model, temperature in ATTEMPTS:
        body = gpt(prompt + feedback, model=model, system=WRITER, temperature=temperature)
        # GPT가 원문을 뒤에 덧붙였으면 잘라낸다
        body = re.split(r"\n\s*(?:\[원래 요청\]|<원래 요청>)", "\n" + body)[0].strip()
        problems = check(body, original, row["task"])
        if not problems:
            break
        feedback = "\n\n이전 출력이 규칙을 어겼다: " + "; ".join(problems) + ". 규칙을 지켜 다시 써라."

    if problems:
        output = None
    else:
        output = f"[과제 유형] {row['task']}\n\n{body}\n\n[원래 요청]\n{original}"
    return {"id": row["id"], "model": model, "body": body, "violations": problems, "output": output}


def make_outputs():
    done = set()
    for r in read_jsonl(OUTPUTS):
        done.add(r["id"])
    todo = []
    for r in read_jsonl(V2_DATASET) + read_jsonl(NEW_INPUTS):
        if r["id"] not in done:
            todo.append(r)

    for result in run_in_parallel(write_answer, todo):
        append_jsonl(OUTPUTS, result)


# ---------- 3. 합치기 ----------
def passthrough(original):
    """고치지 않고 넘기는 예시의 정답"""
    return "[과제 유형] 기타\n\n[원래 요청]\n" + original.strip()


def assemble():
    written = {}
    for r in read_jsonl(OUTPUTS):
        written[r["id"]] = r["output"]

    rows = []
    for row in read_jsonl(V2_DATASET) + read_jsonl(NEW_INPUTS):
        row = dict(row)
        if "source" not in row:
            row["source"] = "v2"

        if row["source"] == "pass":
            row["output"] = passthrough(row["input"])
        elif written.get(row["id"]):
            row["output"] = written[row["id"]]
        else:
            # 규칙 검사를 끝내 통과하지 못한 질문은 넘기는 예시로 바꾼다
            row["output"] = passthrough(row["input"])
            row["source"] += "+check_fail"
        rows.append(row)
    write_jsonl(RAW, rows)


# ---------- 4~6. 보상 모델로 거르기 ----------
ANSWER_MODEL = "qwen2.5:7b"
KINDS = ["orig", "v3", "v3nf"]   # 원문, 정답 그대로, [출력 형식]을 뺀 정답


def is_passthrough(row):
    return row["output"].startswith("[과제 유형] 기타\n\n[원래 요청]")


def drop_output_format(prompt):
    """[출력 형식] 섹션을 다음 섹션 직전까지 지운다."""
    without = re.sub(r"\[출력 형식\].*?(?=\n\s*\[[^\]]+\]|\Z)", "", prompt, flags=re.S)
    return without.replace("\n\n\n", "\n\n")


def prompt_of(row, kind):
    """kind에 맞는 프롬프트 (orig, v3, v3nf)"""
    if kind == "orig":
        return row["input"]
    if kind == "v3":
        return row["output"]
    return drop_output_format(row["output"])


def filter_targets():
    """거를 대상: train이고 넘기는 예시가 아닌 행"""
    rows = []
    for r in read_jsonl(RAW):
        if r["split"] == "train" and not is_passthrough(r):
            rows.append(r)
    return rows


def make_filter_answers():
    done = set()
    for r in read_jsonl(FILTER_ANSWERS):
        done.add((r["id"], r["kind"]))

    for row in filter_targets():
        for kind in KINDS:
            if (row["id"], kind) in done:
                continue
            text = answer(ANSWER_MODEL, [{"role": "user", "content": prompt_of(row, kind)}])
            append_jsonl(FILTER_ANSWERS, {"id": row["id"], "kind": kind, "answer": text})
    unload(ANSWER_MODEL)


def score_filter_answers():
    question = {}
    for r in read_jsonl(RAW):
        question[r["id"]] = r["input"]
    done = set()
    for r in read_jsonl(FILTER_SCORES):
        done.add((r["id"], r["kind"]))

    reward = RewardModel()
    for r in read_jsonl(FILTER_ANSWERS):
        if (r["id"], r["kind"]) in done:
            continue
        score = reward.score(question[r["id"]], r["answer"])
        append_jsonl(FILTER_SCORES, {"id": r["id"], "kind": r["kind"], "score": score})


def apply_filter():
    scores = {}
    for r in read_jsonl(FILTER_SCORES):
        scores[(r["id"], r["kind"])] = r["score"]

    rows = []
    for row in read_jsonl(RAW):
        if row["split"] == "train" and not is_passthrough(row):
            original_score = scores[(row["id"], "orig")]
            v3_score = scores[(row["id"], "v3")]
            v3nf_score = scores[(row["id"], "v3nf")]

            # 두 후보 중 점수가 높은 쪽 (같으면 v3)
            if v3nf_score > v3_score:
                best, best_score = "v3nf", v3nf_score
            else:
                best, best_score = "v3", v3_score

            if best_score <= original_score:
                continue   # 어느 후보도 원문을 못 이겼다

            row["output"] = prompt_of(row, best)
            row["filter"] = {"orig": round(original_score, 3), "v3": round(v3_score, 3),
                             "v3nf": round(v3nf_score, 3), "chosen": best}
        rows.append(row)
    write_jsonl(DATASET, rows)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "inputs":
        make_inputs()
    elif command == "outputs":
        make_outputs()
    elif command == "assemble":
        assemble()
    elif command == "answer":
        make_filter_answers()
    elif command == "score":
        score_filter_answers()
    elif command == "filter":
        apply_filter()
    else:
        print(__doc__)
