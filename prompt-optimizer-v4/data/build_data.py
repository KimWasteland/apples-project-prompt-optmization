"""v4 학습 데이터 만들기.

  python prompt-optimizer-v4/data/build_data.py inputs     # 1. gpt-4o로 짧고 모호한 새 질문 만들기 (16개 주제)
  python prompt-optimizer-v4/data/build_data.py guides     # 2. gpt-4o-mini가 질문마다 답변 가이드 쓰기
  python prompt-optimizer-v4/data/build_data.py rewrite    # 3. gpt-4o-mini가 가이드를 녹여 질문을 한 문단으로 다시 쓰기
  python prompt-optimizer-v4/data/build_data.py assemble   # 4. dataset_v4.jsonl

대상: v3 데이터의 질문과 새 질문 중 gate()를 통과한 것 (../optimizer.py)
정답: 다시 쓴 한 문단 (검사 통과), 또는 원래 질문 그대로 (정답이 하나인 요청, 요약할 글이 없는 요약 요청)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 버전 폴더의 utils.py, optimizer.py를 불러오기 위해

from optimizer import SYSTEM, faithful, gate, tone_of
from utils import (DATA, KOMT_QUESTIONS, V3_RAW, WILDCHAT_QUESTIONS, gpt, is_similar_to_any, read_jsonl,
                   run_in_parallel, write_jsonl)

NEW_INPUTS = DATA / "build" / "new_inputs.jsonl"
GUIDES = DATA / "build" / "guides.jsonl"
REWRITES = DATA / "build" / "rewrites.jsonl"
DATASET = DATA / "dataset_v4.jsonl"

KEEP = "그대로"   # 가이드 자리에 이 글자가 있으면 원래 질문을 그대로 정답으로 쓴다


def all_inputs():
    """v3 데이터의 질문 + 새로 만든 질문"""
    return read_jsonl(V3_RAW) + read_jsonl(NEW_INPUTS)


# ---------- 1. 새 질문 ----------
INPUT_GEN = """한국어 사용자가 AI 챗봇에게 실제로 보낼 법한 짧은 첫 메시지를 {n}개 만들어라. 주제: {topic}
- 길이는 5~50자. 대부분은 정보가 부족한 짧고 모호한 요청 (예: "다이어트 식단 추천", "엑셀 vlookup 뭐임", "자소서 첫 문장 어케 써").
- 말투를 섞는다: 반말, 존댓말, 구어체, 줄임말, 영어 혼용. 오타도 가끔.
- 질문, 추천, 비교, 설명, 계획, 글쓰기(창작·편지·글귀), 아이디어 요청을 섞는다. 숫자 계산 문제나 분량·형식 지정은 넣지 않는다.
- 서로 겹치지 않게.
JSON으로만: {{"requests": [{{"text": "...", "task": "설명|추천|기획|비교|요약|기타"}}, ...]}}"""
TOPICS = ["일상·생활 팁", "요리·음식", "건강·운동·다이어트", "돈·재테크·보험", "법·행정·세금", "연애·인간관계·가족",
          "육아·교육", "공부·시험·자격증", "취업·이직·직장생활", "IT·프로그래밍 개념과 도구", "글쓰기·창작·SNS 문구",
          "여행·지역", "쇼핑·제품 추천", "취미·게임·문화·엔터", "과학·역사·상식", "반려동물·자동차·집"]


def make_inputs():
    # 평가 문항(WildChat 300, KoMT)이나 기존 질문과 비슷한 것은 뺀다
    avoid = []
    for r in read_jsonl(V3_RAW):
        avoid.append(r["input"])
    for r in read_jsonl(WILDCHAT_QUESTIONS):
        avoid.append(r["q"])
    for q in read_jsonl(KOMT_QUESTIONS):
        avoid.append(q["turns"][0])

    rows = []
    for topic in TOPICS:
        reply = gpt(INPUT_GEN.format(n=40, topic=topic), model="gpt-4o-2024-08-06", temperature=0.9, json_mode=True)
        for item in json.loads(reply)["requests"]:
            text = item["text"].strip()
            if not gate(text) or is_similar_to_any(text, avoid):
                continue
            avoid.append(text)
            rows.append({"id": f"v4n_{len(rows) + 1:04d}", "source": "v4new", "task": item.get("task", "기타"),
                         "level": "L1", "input": text, "split": "train", "topic": topic})
    write_jsonl(NEW_INPUTS, rows)


# ---------- 2. 가이드 ----------
TEACHER = """너는 '답변 가이드' 작성기다. 사용자의 짧은 요청에 답하지 말고, 다른 AI가 이 요청에 더 좋은 답을 하도록 이 분야를 잘 아는 사람이 짚어 줄 내용을 써라.
- 갈림길: 요청이 실제로 모호해서 답이 상황에 따라 크게 달라지면, 그 기준을 한 줄로 (예: "용도(개발/문서 작업/휴대성)에 따라 추천이 달라짐"). 모호하지 않으면 빈 문자열.
- 다룰 내용: 이 요청에만 해당하는 핵심 고려사항 2~4개. 짧은 명사구로.
규칙:
1. 다룰 내용은 이 요청에서만 나오는 구체적인 것이어야 한다. "장단점", "비용", "기능", "정의", "절차", "예시", "주의사항", "사용자 경험"처럼 어떤 요청에나 붙는 말은 그 자체로 쓰지 말고, 무엇의 비용·위험인지 구체적으로 쓴다 (예: "비용" 대신 "보증금 반환 위험", "대출 이자와 월세 부담 비교").
2. 널리 알려지고 확실한 사실에 근거한 고려사항만 쓴다. 불확실한 수치, 날짜, 특정 제품·회사 이름, 사용자 정보(나이, 직업, 상황)는 만들지 않는다.
3. 요청에 이미 있는 조건을 바꾸지 않고, 분량·톤·형식을 정하지 않는다.
4. 요청과 같은 언어로 쓴다.
5. 다음 경우만 가이드가 필요 없다: 정답이 하나로 정해진 계산·수학 문제·퍼즐, 요청 안에 주어진 코드·글·데이터를 처리하는 요청(오류 찾기, 정보 뽑기), 요약·검토할 글이 요청에 없는 요약 요청. 이때는 {"그대로": true}만 출력한다. 개념·정의 질문("~가 뭐야", "~ 설명해줘")이나 코드 작성 요청은 여기에 해당하지 않으니 가이드를 쓴다.
JSON으로만: {"갈림길": "...", "다룰 내용": ["...", "..."]}"""
FORCE = "\n(이 요청은 가이드가 필요하다. 그대로를 출력하지 마라.)"

# 어디에나 붙는 말. 다룰 내용이 이것뿐이면 안 된다
GENERIC = ["장단점", "비용", "기능", "정의", "절차", "예시", "주의사항", "주의할 점", "사용자 경험", "가격", "성능", "특징",
           "차이점", "선택 기준", "용도", "목적", "효과", "방법", "종류"]

# GPT의 "그대로" 판단은 비슷한 질문끼리도 엇갈려서 규칙으로 고정한다.
# 코드 작성이나 개념 질문은 GPT가 "그대로"라고 해도 가이드를 다시 받는다. 단, 고칠 코드나 글이 주어진 질문은 예외.
WRITE_OR_CONCEPT = re.compile(r"만들어|짜\s*줘|짜주|작성|구현|코드\s*(?:ㄱ|좀|줘)|뭐야|뭐임|뭔가요|무엇|설명|알려|차이|퀴즈|추천")
GIVEN_MATERIAL = re.compile(r"왜|에러|오류|고쳐|안 돌아|이\s*(?:코드|글|텍스트|문장)|다음|아래|:")


def should_keep(row, guide):
    """원래 질문을 그대로 정답으로 쓸 질문인가"""
    if row["task"] == "요약":
        return True   # 짧은 질문에는 요약할 글이 없다
    if guide != KEEP:
        return False
    # GPT가 "그대로"라고 했어도, 코드 작성이나 개념 질문이면 가이드를 쓴다
    question = row["input"]
    if WRITE_OR_CONCEPT.search(question) and not GIVEN_MATERIAL.search(question):
        return False
    return True


def english_words(text):
    """두 글자 이상 영어 단어 (소문자)"""
    return {word.lower() for word in re.findall(r"[A-Za-z]{2,}", text)}


def check_guide(original, fork, points):
    """가이드의 문제 목록. 빈 목록이면 통과."""
    text = fork + " " + " ".join(points)
    problems = []

    if len(points) < 2 or len(points) > 4:
        problems.append("aspects")   # 다룰 내용은 2~4개

    generic = [g.replace(" ", "") for g in GENERIC]
    for point in points:
        if point.replace(" ", "") in generic:
            problems.append("generic")   # 어디에나 붙는 말
            break

    if set(re.findall(r"\d+", text)) - set(re.findall(r"\d+", original)):
        problems.append("new_number")   # 원래 질문에 없는 숫자
    if english_words(text) - english_words(original):
        problems.append("new_latin")   # 원래 질문에 없는 영어 단어 (제품, 회사 이름 등)
    if re.search(r"[一-鿿぀-ヿ]", text):
        problems.append("cjk")   # 한자, 일본 글자
    if len(text) > 120:
        problems.append("too_long")
    return problems


def ask_guide(question, force=False):
    """GPT에 가이드를 받는다. (가이드 글, 문제 목록). "그대로"면 (KEEP, [])."""
    prompt = question
    if force:
        prompt += FORCE
    reply = json.loads(gpt(prompt, system=TEACHER, json_mode=True))
    if reply.get(KEEP):
        return KEEP, []

    fork = (reply.get("갈림길") or "").strip()
    points = [p.strip() for p in reply.get("다룰 내용", [])]
    guide = ""
    if fork:
        guide += f"갈림길: {fork}\n"
    guide += "다룰 내용: " + " / ".join(points)
    return guide, check_guide(question, fork, points)


def make_guide(row):
    guide, problems = ask_guide(row["input"])
    # GPT가 "그대로"라고 했지만 규칙상 가이드가 필요하면 한 번 더 묻는다
    if guide == KEEP and not should_keep(row, guide):
        guide, problems = ask_guide(row["input"], force=True)
    return {"id": row["id"], "input": row["input"], "task": row["task"], "guide": guide, "bad": problems}


def make_guides():
    # train과 val만 만든다. val은 학습 중 loss 확인용이다
    todo = []
    for r in all_inputs():
        if r["split"] in ("train", "val") and gate(r["input"]):
            todo.append(r)
    write_jsonl(GUIDES, run_in_parallel(make_guide, todo))


# ---------- 3. 다시 쓰기 ----------
REWRITE = """사용자가 AI에게 보내려는 짧은 <요청>과, 좋은 답을 위해 짚어야 할 <가이드>가 있다.
<요청>을 한 문단의 프롬프트로 다시 써라. 요청에 답하지 마라.
규칙:
1. 원래 요청의 모든 정보(대상, 상황, 조건, 용어, 숫자)를 하나도 빼지 않는다. 원래 요청의 표현을 최대한 그대로 쓰고, 영어 용어는 영어 그대로 둔다.
2. 말투를 원래 요청과 똑같이 맞춘다. 원래 요청이 존댓말이면 모든 문장을 존댓말로, 반말이면 모든 문장을 반말로.
3. 가이드의 다룰 내용과 갈림길을 사용자가 직접 부탁하는 문장으로 자연스럽게 녹인다. 갈림길이 요청과 맞지 않으면 버린다.
4. 가이드에 없는 내용(구체적인 예, 방법, 사실)과 인사말, "부탁해" 같은 군더더기, 분량·형식 지시는 더하지 않는다.
5. 헤더·목록·괄호 없이 2~4문장의 한 문단. 각 문장은 마침표나 물음표로 깔끔하게 끝낸다. 다시 쓴 프롬프트만 출력한다.
<요청>{q}</요청>
<가이드>{g}</가이드>"""
CASUAL_HINT = "\n주의: 원래 요청은 반말이다. 모든 문장을 반말(~줘, ~해, ~야, ~어?)로 끝내고 '~요', '~니다', '~세요'를 절대 쓰지 않는다."
POLITE_HINT = "\n주의: 모든 문장을 존댓말(~요, ~니다)로 끝낸다."


def rewrite(guide_row):
    """다시 쓰고 faithful()로 검사한다. 문제가 있으면 temperature 0.7로 한 번 더 쓴다."""
    question = guide_row["input"]
    if tone_of(question) == "casual":
        hint = CASUAL_HINT
    else:
        hint = POLITE_HINT

    for temperature in [0.0, 0.7]:
        text = gpt(REWRITE.format(q=question, g=guide_row["guide"]) + hint, temperature=temperature)
        problems = faithful(question, text)
        if not problems:
            break
    return {"id": guide_row["id"], "input": question, "output": text, "bad": problems}


def make_rewrites():
    rows = {}
    for r in all_inputs():
        rows[r["id"]] = r

    todo = []
    for g in read_jsonl(GUIDES):
        if g["guide"] == KEEP or g["bad"]:
            continue
        if should_keep(rows[g["id"]], g["guide"]):
            continue
        todo.append(g)
    write_jsonl(REWRITES, run_in_parallel(rewrite, todo))


# ---------- 4. 합치기 ----------
# 다시 쓴 결과에서 추가로 빼는 경우
MADE_UP_ASK = re.compile(r"얼마|수수료|일정|정확도|최신|하더라고요|달라진다고|연락처")   # 수치나 최신 정보를 지어내게 하는 부탁
ROLE_SLIP = re.compile(r"(고려|생각)해\s*봐|생각해\s*봐야|가장 중요해\?|좋고, .*나을 수 있어")   # 부탁이 아니라 답을 쓰기 시작함


def usable_rewrite(question, text):
    if faithful(question, text):
        return False
    if ROLE_SLIP.search(text):
        return False
    if MADE_UP_ASK.search(text) and not MADE_UP_ASK.search(question):
        return False
    return True


def assemble():
    guides = {}
    for r in read_jsonl(GUIDES):
        guides[r["id"]] = r["guide"]
    rewrites = {}
    for r in read_jsonl(REWRITES):
        rewrites[r["id"]] = r["output"]

    rows = []
    for row in all_inputs():
        if not gate(row["input"]):
            continue
        new = {"id": row["id"], "task": row["task"], "level": row["level"], "input": row["input"],
               "split": row["split"], "system": SYSTEM}

        if row["split"] == "test":
            new["output"] = ""   # test는 train.py가 생성 결과를 볼 때만 쓴다
        elif should_keep(row, guides.get(row["id"])):
            new["output"] = row["input"]
        elif row["id"] in rewrites and usable_rewrite(row["input"], rewrites[row["id"]]):
            new["output"] = rewrites[row["id"]]
        else:
            continue
        rows.append(new)
    write_jsonl(DATASET, rows)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "inputs":
        make_inputs()
    elif command == "guides":
        make_guides()
    elif command == "rewrite":
        make_rewrites()
    elif command == "assemble":
        assemble()
    else:
        print(__doc__)
