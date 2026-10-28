"""SLM이 생성한 라벨 JSON + 원래 질문 -> 최종 최적화 프롬프트.

사용:
  python build_prompt.py "질문"                      # 학습된 모델로 라벨 생성 후 프롬프트 출력
  python build_prompt.py "질문" --adapter models/lora_v3
  python build_prompt.py --preds results/preds/ood2_preds_v3.jsonl # 평가 결과 파일의 예측 JSON들을 프롬프트로 변환
"""
import argparse, json, re
from prep import CATALOG, SYSTEM

# ------------------------------------------------------------------ 카탈로그 (데이터셋 포맷.md)
FORMAT_PROMPT_CATALOG = {
    # 요약
    "F_SUM_BULLETS": "핵심 내용만 추출하여 기호(-)를 사용한 글머리 기호 형태의 문장으로 작성하세요.",
    "F_SUM_PROSE": "기호나 번호 없이 자연스럽게 이어지는 1~2개의 완결된 단락(줄글)으로 작성하세요.",
    "F_SUM_KEYWORD": "주요 핵심 개념을 **키워드**: 한 줄 설명 구조의 목록으로 작성하세요.",
    "F_SUM_TIMELINE": "시간적/논리적 전개 순서에 따라 번호 매기기(1, 2, 3...) 구조로 작성하세요.",
    # 비교
    "F_COMP_TABLE": "여러 대상과 비교 기준(속성)을 행과 열로 교차 배치한 마크다운 표([비교 기준 | 대상 A | 대상 B]) 형식으로 작성하세요.",
    "F_COMP_PRO_CON": "각 대상을 중심으로 장점과 단점 또는 특징 및 실익을 구별하여 목록화된 형태로 작성하세요.",
    "F_COMP_VENN": "공통점 영역과 핵심 차이점 영역을 명확히 분리하여 이원 구조로 작성하세요.",
    # 설명
    "F_EXP_STEPWISE": "개념과 동작 원리를 단계별(1단계, 2단계...)로 차근차근 설명하세요.",
    "F_EXP_QA": "핵심 질문(Q)과 그에 대한 명쾌한 답변(A) 구조를 반복하여 주요 의문점을 풀어주는 형태로 작성하세요.",
    "F_EXP_DEEPDIVE": "개념의 정의, 핵심 구성 요소, 실제 작동 메커니즘, 파급 효과까지 다각도로 깊이 있게 설명하세요.",
    # 추천
    "F_REC_TOPN": "가장 추천하는 항목 N개를 선정하고, 각 항목별 핵심 추천 이유와 특징을 순위/목록 형태로 정리하세요.",
    "F_REC_SITUATIONAL": "사용자의 다양한 상황, 예산, 사용 목적 등 조건별(예: '초보자용', '가성비 위주') 최적의 선택지를 분류하여 추천하세요.",
    "F_REC_PRO_CON": "추천하는 각 항목의 주요 특징과 함께 선택 시 고려해야 할 장점과 단점(주의사항)을 명확히 명시하여 추천하세요.",
    "F_REC_SCORECARD": "추천 항목들을 주요 평가 기준(가격, 성능, 편의성 등)에 따라 항목별 점수 또는 등급으로 매겨 비교 추천하세요.",
    # 기획
    "F_PLAN_IDEA_LIST": "아이디어의 명칭, 핵심 컨셉, 기대 효과를 한 세트로 묶어 3~5개의 구체적인 아이디어를 제안하세요.",
    "F_PLAN_SWOT": "기획안의 내부 요인(강점, 약점)과 외부 환경(기회, 위협)을 분석하여 전략적 방향성을 제시하세요.",
    "F_PLAN_FRAMEWORK": "문제 정의, 해결 방안, 실행 단계(Action Item), 예상 리스크 및 대응책의 구조화된 프레임워크로 작성하세요.",
    "F_PLAN_MINDMAP": "대주제를 중심으로 중주제와 세부 실행 아이디어가 가지를 치듯 확장되는 계층적 구조로 작성하세요.",
}

CONSTRAINT_PROMPT_CATALOG = {
    # 요약
    "C_SUM_LENGTH_SHORT": "전체 요약 분량을 공백 포함 300자 이내(또는 3문장 이내)로 짧고 압축적으로 작성하세요.",
    "C_SUM_LENGTH_LONG": "주요 핵심 논거와 세부 맥락이 누락되지 않도록 800자 이상의 충분한 분량으로 상세히 작성하세요.",
    "C_SUM_NO_JARGON": "전문 용어나 어려운 기술적 단어를 배제하고, 초등학생도 이해할 수 있는 쉬운 어휘로 풀어서 작성하세요.",
    "C_SUM_FACT_ONLY": "원문 분석자의 추측이나 주관적 해석을 완전히 배제하고, 원문에 명시된 객관적 수치와 팩트만 남겨 작성하세요.",
    "C_SUM_ACTION_ITEM": "전체 내용 중 향후 조치해야 할 실행 과제(Action Item) 및 후속 조치 사항 중심으로 강조하여 작성하세요.",
    # 비교
    "C_COMP_BALANCED": "어느 한쪽으로 편향되지 않도록 모든 비교 대상의 장단점과 비중을 중립적이고 균형 있게 작성하세요.",
    "C_COMP_VERDICT": "단순 비교에 그치지 않고, 각 상황이나 사용자 유형에 따라 최종적으로 어떤 대상이 더 우수한지 결론(Verdict)을 제시하세요.",
    "C_COMP_DIFF_ONLY": "공통적인 일반 기능은 생략하고, 두 대상 간의 결정적인 차이점(Differentiator) 및 핵심 차별화 요소만 집중 작성하세요.",
    "C_COMP_COST_EFFICIENCY": "성능, 가격, 유효 기간, 유지비 등 비용 대비 효과(가성비 및 ROI) 측면의 기준을 가장 최우선으로 두어 비교하세요.",
    "C_COMP_SPECIFIC_METRIC": "입력 질문에서 언급된 특정 기준(예: 보안성, 속도, 편의성 등)에 초점을 맞춰 해당 항목 위주로 깊이 있게 비교하세요.",
    # 설명
    "C_EXP_EASY_ELI5": "전문 용어를 최소화하고 쉬운 일상 어휘와 비유를 활용하여, 어린이나 초보자도 이해할 수 있도록 쉽게 작성하세요.",
    "C_EXP_TECHNICAL_DEEP": "해당 분야의 전문 용어와 정확한 메커니즘을 명확히 사용하여 전공자/전문가 수준의 깊이 있고 엄밀한 표현으로 작성하세요.",
    "C_EXP_VISUAL_ASCII": "텍스트 설명과 함께 이해를 돕기 위한 텍스트 다이어그램(ASCII 아트, 텍스트 흐름도 등)을 포함하여 시각적으로 작성하세요.",
    "C_EXP_REAL_WORLD_CASE": "추상적인 개념 설명에 그치지 않고, 실제로 적용되는 구체적인 실제 사례(Use Case)나 산업 적용 예시를 반드시 포함하세요.",
    "C_EXP_MISCONCEPTION": "해당 개념과 관련하여 사람들이 흔히 오해하거나 착각하기 쉬운 지점(Common Pitfalls)을 짚어주고 바르게 정정하여 작성하세요.",
    # 추천
    "C_REC_VALUE_FOR_MONEY": "성능, 품질, 기능 대비 가격의 합리성(가성비)을 최우선 평가 기준으로 설정하여 추천하세요.",
    "C_REC_STRICT_CRITERIA": "추천 대상 선정 시 기준이 된 분명한 평가 항목(예: 성능, 디자인, 편의성 등)과 선정이유의 객관적 근거를 명확히 제시하세요.",
    "C_REC_USER_PERSONA": "사용자의 특정 상황(예: 입문자, 독거 가구, 헤비 유저 등)이나 페르소나에 부합하는 최적의 이유를 부각하여 추천하세요.",
    "C_REC_ALTERNATIVES": "1순위 최선책(Top Pick) 외에도 대체 가능한 차선책(Alternative)이나 저렴한 옵션을 함께 명시하여 추천하세요.",
    "C_REC_CAUTION_NOTE": "단순 추천에 그치지 않고, 해당 선택 시 발생할 수 있는 리스크, 단점, 구매 전 유의사항(Caveats)을 반드시 명시하세요.",
    # 기획
    "C_PLAN_FEASIBILITY": "현실적인 예산, 인력, 기술적 제약(Feasibility)을 최우선으로 고려하여 즉시 실행 가능한 방안 위주로 작성하세요.",
    "C_PLAN_CREATIVE_MOONSHOT": "기존의 전형적인 방식이나 틀을 깨고, 파격적이고 독창적인 아이디어(Out-of-the-box)에 초점을 맞춰 제안하세요.",
    "C_PLAN_KPI_MEASURABLE": "아이디어 제안에 그치지 않고, 성공 여부를 측정할 수 있는 구체적인 정량적 성과 지표(KPI)와 목표 수치를 함께 제시하세요.",
    "C_PLAN_STEP_BY_STEP": "아이디어 기획부터 준비, 실행, 검토까지 시간 순서에 따른 단계별 로드맵(Roadmap)과 핵심 액션 아이템을 포함하세요.",
    "C_PLAN_RISK_MITIGATION": "기획안 실행 시 발생할 수 있는 예상 문제점, 법적/운영상 리스크 및 이에 대한 구체적인 대응 전략(Plan B)을 함께 명시하세요.",
}

TASK_PROMPT = {
    "summarization": "사용자가 제공한 내용을 요약하세요.",
    "comparison": "사용자가 언급한 대상들을 비교 분석하세요.",
    "explanation": "사용자가 질문한 개념을 설명하세요.",
    "recommendation": "사용자의 요청에 맞는 항목을 추천하세요.",
    "planning": "사용자의 요청에 맞는 기획안과 아이디어를 제안하세요.",
}

assert set(FORMAT_PROMPT_CATALOG) == {f for fs, _ in CATALOG.values() for f in fs}
assert set(CONSTRAINT_PROMPT_CATALOG) == {c for _, cs in CATALOG.values() for c in cs}

# 명세상 디폴트 constraint 중 일부는 중립적이지 않다 (설명=어린이 눈높이+비유, 추천=가성비 최우선).
# 모델이 디폴트를 골랐는데 질문에 해당 요청 표현이 없으면 = "조건 미지정"으로 보고 완화된 일반 지시로 바꾼다.
SOFT_DEFAULT = {
    "C_EXP_EASY_ELI5": ("C_EXP_GENERAL",
                        r"쉽|초등|초보|입문|어린|아이(가|한테|에게|들|도|용|처럼)|애한테|애들|꼬마|비유|비전공|문외한|일반인|"
                        r"모르|알아듣|알아들|할머니|할아버지|부모님|노베이스|눈높이|ELI5|"
                        r"\d+\s*살|[한두세네]\s*살|다섯\s*살|여섯\s*살|일곱\s*살|조카|아들|딸(한테|에게|이)|유치원|초딩|중학생"),
    "C_REC_VALUE_FOR_MONEY": ("C_REC_GENERAL",
                              r"가성비|가심비|가격 ?대비|저렴|(?<!비)싼|(?<!비)싸(고|게|면|도)|합리적|최저가|예산|"
                              r"\d+\s*만\s*원|원 ?(이하|이내|안쪽|대)"),
}
CONSTRAINT_PROMPT_CATALOG["C_EXP_GENERAL"] = "전문 용어는 처음 나올 때 짧게 풀이하고, 핵심 개념이 명확히 전달되도록 정확하고 이해하기 쉽게 작성하세요."
CONSTRAINT_PROMPT_CATALOG["C_REC_GENERAL"] = "가격, 성능, 사용 편의성 등 주요 기준을 균형 있게 고려하여 추천하세요."


def parse_label(text):
    """모델 출력 문자열 -> dict. 코드펜스나 앞뒤 잡음이 있어도 첫 JSON 객체를 꺼낸다."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def normalize(label, query=None):
    """카탈로그를 벗어난 값을 보정: task -> format/constraint ID 접두사로 추론 -> 실패 시 explanation.
    format/constraint가 task 목록에 없으면 해당 task의 디폴트로 대체.
    query를 주면 SOFT_DEFAULT 규칙도 적용한다."""
    fixes = []
    task = label.get("task")
    if task not in CATALOG:
        prefix = {"SUM": "summarization", "COMP": "comparison", "EXP": "explanation", "REC": "recommendation", "PLAN": "planning"}
        guess = str(label.get("format", "")).split("_")[1:2]
        task = prefix.get(guess[0] if guess else "", "explanation")
        fixes.append(f"task {label.get('task')!r} -> {task}")
    fids, cids = CATALOG[task]
    if cids[0] in SOFT_DEFAULT:  # 이미 완화된 라벨을 다시 넣어도 유지
        cids = cids + [SOFT_DEFAULT[cids[0]][0]]
    fmt = label.get("format") if label.get("format") in fids else fids[0]
    con = label.get("constraint") if label.get("constraint") in cids else cids[0]
    if fmt != label.get("format"):
        fixes.append(f"format {label.get('format')!r} -> {fmt}")
    if con != label.get("constraint"):
        fixes.append(f"constraint {label.get('constraint')!r} -> {con}")
    if query is not None and con in SOFT_DEFAULT and not re.search(SOFT_DEFAULT[con][1], query):
        fixes.append(f"constraint {con} -> {SOFT_DEFAULT[con][0]} (요청에 해당 조건 없음)")
        con = SOFT_DEFAULT[con][0]
    return {
        "role": label.get("role") or "도움이 되는 AI 어시스턴트",
        "purpose": label.get("purpose") or TASK_PROMPT[task],
        "task": task, "format": fmt, "constraint": con,
    }, fixes


# ① 카탈로그 문구는 고정값(300자, 800자, N개 등)이라 사용자가 명시한 조건과 충돌할 수 있음 -> 사용자 요청 우선
PRIORITY_RULE = (
    "위 지침은 기본값입니다. 사용자 요청에 명시된 숫자(분량, 개수, 예산 등), 대상, 평가 기준, "
    "추가 조건이 위 지침과 다르거나 지침에 없으면 사용자 요청을 우선하여 모두 반영하세요."
)

FENCE = re.compile(r"^\s*(\"{3}|'{3}|```)\w*\s*$\n(.*?)\n^\s*\1\s*$", re.M | re.S)
SEPARATORS = re.compile(r"^\s*(?:-{3,}|={3,})\s*$", re.M)


def split_request(text, max_request=80, min_body=100):
    """② 요청 문장과 붙여넣은 본문(요약할 글 등)을 분리. 분류 모델에는 요청만 넣는다.
    1) 따옴표 블록(\"\"\", ''', ```)이 있으면: 블록 안 = 본문, 블록 밖 = 요청
    2) 구분선(---, ===)이 있으면: 가장 짧은 덩어리 = 요청, 나머지 = 본문
    3) 구분선 없는 여러 줄: 첫 줄/마지막 줄 중 짧은 쪽이 max_request자 이하이고
       나머지가 min_body자 이상이면서 그 2배 이상 길면 그 줄이 요청 (짧은 여러 줄 요청은 분리 안 함)
    4) 그 외: 전체가 요청, 본문 없음"""
    text = text.strip()
    blocks = [m.group(2).strip() for m in FENCE.finditer(text)]
    outside = FENCE.sub("", text).strip()
    if blocks and outside:
        return outside, "\n\n".join(blocks)
    parts = [p.strip() for p in SEPARATORS.split(text) if p.strip()]
    if len(parts) >= 2:
        req = min(parts, key=len)
        return req, "\n\n".join(p for p in parts if p is not req)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        first_is_req = len(lines[0]) <= len(lines[-1])
        req = lines[0] if first_is_req else lines[-1]
        body = "\n".join(lines[1:] if first_is_req else lines[:-1])
        if len(req) <= max_request and len(body) >= max(min_body, 2 * len(req)):
            return req, body
    return text, ""


def build_prompt(query, label, content=""):
    lb, fixes = normalize(label, query)
    prompt = (
        f"# 역할\n당신은 {lb['role']}입니다.\n\n"
        f"# 목적\n{lb['purpose']}\n\n"
        f"# 작업\n{TASK_PROMPT[lb['task']]}\n\n"
        f"# 출력 형식\n{FORMAT_PROMPT_CATALOG[lb['format']]}\n\n"
        f"# 제약 조건\n{CONSTRAINT_PROMPT_CATALOG[lb['constraint']]}\n\n"
        f"# 우선순위\n{PRIORITY_RULE}\n\n"
        f"# 사용자 요청\n{query}"
    )
    if content:
        prompt += f"\n\n# 입력 자료\n<<<\n{content}\n>>>"
    return prompt, lb, fixes


class Labeler:
    """학습된 SLM으로 질문 -> 라벨 JSON."""

    def __init__(self, adapter, base="Qwen/Qwen2.5-3B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
        self.model = PeftModel.from_pretrained(model, adapter).eval()

    def __call__(self, query):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(0)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=160, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def show(query, raw, content=""):
    prompt, lb, fixes = build_prompt(query, parse_label(raw), content)
    print("=" * 70)
    print("[라벨]", json.dumps(lb, ensure_ascii=False))
    if fixes:
        print("[보정]", "; ".join(fixes))
    print("-" * 70)
    print(prompt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--adapter", default="models/lora_v3")
    ap.add_argument("--preds", help="evaluate.py가 만든 *_preds_*.jsonl")
    ap.add_argument("--content-file", help="요약할 글 등 본문 파일 (분류에는 쓰지 않고 프롬프트에만 첨부)")
    ap.add_argument("--file", help="요청+본문이 섞인 텍스트 파일 (split_request로 자동 분리)")
    ap.add_argument("--out", help="결과를 jsonl로 저장")
    a = ap.parse_args()

    results = []  # (request, raw model output, content)
    if a.preds:
        for l in open(a.preds, encoding="utf-8"):
            r = json.loads(l)
            show(r["query"], r["pred"])
            results.append((r["query"], r["pred"], ""))
    else:
        labeler = Labeler(a.adapter)
        inputs = []
        if a.file:
            inputs.append(split_request(open(a.file, encoding="utf-8").read()))
        elif a.query:
            req, body = split_request(" ".join(a.query))
            if a.content_file:
                body = (body + "\n\n" if body else "") + open(a.content_file, encoding="utf-8").read().strip()
            inputs.append((req, body))
        else:  # 인자 없으면 대화형
            print("질문을 입력하세요 (빈 줄 입력 시 종료)")
            while (q := input("> ").strip()):
                req, body = split_request(q)
                raw = labeler(req); show(req, raw, body); results.append((req, raw, body))
        for req, body in inputs:
            raw = labeler(req); show(req, raw, body); results.append((req, raw, body))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fp:
            for q, raw, body in results:
                prompt, lb, fixes = build_prompt(q, parse_label(raw), body)
                fp.write(json.dumps({"query": q, "content": body, "label": lb, "fixes": fixes, "prompt": prompt}, ensure_ascii=False) + "\n")
