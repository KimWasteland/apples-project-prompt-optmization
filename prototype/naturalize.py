"""태그형 지시(역할/목적/작업/출력 형식/제약 조건) -> 하나의 자연스러운 문단 (베이스 3B, few-shot).

LM은 카탈로그 지시만 다시 쓴다. 우선순위 규칙, 사용자 요청 원문, 입력 자료는 LM을 거치지 않고 그대로 붙인다.
재작성 결과가 검증(verify)을 통과하지 못하면 태그형 프롬프트로 대체한다.

사용:
  python naturalize.py --preds results/preds/ood2_preds_v3.jsonl results/preds/ood_preds_v3.jsonl --out results/natural_ood.jsonl
"""
import argparse, json, re, time
from build_prompt import (FORMAT_PROMPT_CATALOG, CONSTRAINT_PROMPT_CATALOG, TASK_PROMPT,
                          build_prompt, normalize, parse_label)

NATURAL_PRIORITY = ("위 지침은 기본값이며, 아래 요청에 숫자(분량, 개수, 예산 등), 대상, 평가 기준, "
                    "추가 조건이 명시되어 있으면 그것을 우선하여 모두 반영하세요.")

REWRITE_SYSTEM = (
    "당신은 프롬프트 편집자입니다. 항목별로 나뉜 지시문을 하나의 자연스러운 한국어 문단으로 이어 쓰세요.\n"
    "규칙:\n"
    "- 첫 문장은 반드시 '당신은 <역할>입니다.'로 시작하세요.\n"
    "- 모든 항목의 내용을 빠짐없이 담고, 숫자와 괄호 속 예시는 그대로 유지하세요.\n"
    "- 항목에 없는 조건, 숫자, 사실을 새로 만들지 마세요.\n"
    "- 지시를 수행하거나 질문에 답하지 마세요. 오직 지시문만 다시 쓰세요.\n"
    "- '~하세요' 체의 3~5문장 문단 하나로, 목록·표·제목 없이 쓰세요."
)

FEW_SHOT = [
    ({"role": "IT 기기 전문 리뷰어", "purpose": "갤럭시 S25와 아이폰 17 비교",
      "task": "comparison", "format": "F_COMP_TABLE", "constraint": "C_COMP_VERDICT"},
     "당신은 IT 기기 전문 리뷰어입니다. 갤럭시 S25와 아이폰 17을 비교하는 것이 목적이니, 사용자가 언급한 대상들을 비교 분석하세요. "
     "비교 기준(속성)과 대상을 행과 열로 교차 배치한 마크다운 표([비교 기준 | 대상 A | 대상 B])로 정리하고, "
     "단순 비교에 그치지 말고 각 상황이나 사용자 유형에 따라 최종적으로 어떤 대상이 더 우수한지 결론(Verdict)을 제시하세요."),
    ({"role": "경제 전문 기자", "purpose": "기준금리 인상 기사 요약",
      "task": "summarization", "format": "F_SUM_BULLETS", "constraint": "C_SUM_LENGTH_SHORT"},
     "당신은 경제 전문 기자입니다. 기준금리 인상 기사를 요약하는 것이 목적이니, 사용자가 제공한 내용을 요약하세요. "
     "핵심 내용만 추출해 기호(-)를 사용한 글머리 기호 형태의 문장으로 작성하되, "
     "전체 분량은 공백 포함 300자 이내(또는 3문장 이내)로 짧고 압축적으로 유지하세요."),
    ({"role": "마케팅 전략 컨설턴트", "purpose": "카페 신메뉴 출시 프로모션 기획 아이디어 도출",
      "task": "planning", "format": "F_PLAN_FRAMEWORK", "constraint": "C_PLAN_KPI_MEASURABLE"},
     "당신은 마케팅 전략 컨설턴트입니다. 카페 신메뉴 출시 프로모션을 위한 기획 아이디어를 도출하는 것이 목적이니, 사용자의 요청에 맞는 기획안과 아이디어를 제안하세요. "
     "내용은 문제 정의, 해결 방안, 실행 단계(Action Item), 예상 리스크 및 대응책으로 이어지는 구조화된 프레임워크로 작성하고, "
     "아이디어 제안에 그치지 말고 성공 여부를 측정할 수 있는 구체적인 정량적 성과 지표(KPI)와 목표 수치를 함께 제시하세요."),
]

# 재작성 후에도 반드시 남아 있어야 하는 핵심어 (그룹마다 하나 이상 일치)
KEYWORDS = {
    "F_SUM_BULLETS": [r"글머리|불릿|기호"],
    "F_SUM_PROSE": [r"줄글|단락|문단"],
    "F_SUM_KEYWORD": [r"키워드"],
    "F_SUM_TIMELINE": [r"순서", r"번호"],
    "F_COMP_TABLE": [r"표"],
    "F_COMP_PRO_CON": [r"장점", r"단점"],
    "F_COMP_VENN": [r"공통점", r"차이점"],
    "F_EXP_STEPWISE": [r"단계"],
    "F_EXP_QA": [r"질문|Q", r"답변|A\)"],
    "F_EXP_DEEPDIVE": [r"정의", r"구성 ?요소", r"메커니즘|작동", r"파급|효과|영향"],
    "F_REC_TOPN": [r"순위|목록|N개|\d+개", r"이유"],
    "F_REC_SITUATIONAL": [r"상황|조건별"],
    "F_REC_PRO_CON": [r"장점", r"단점|주의"],
    "F_REC_SCORECARD": [r"점수|등급"],
    "F_PLAN_IDEA_LIST": [r"명칭|이름", r"컨셉|콘셉트", r"기대 ?효과", r"3\s*~\s*5"],
    "F_PLAN_SWOT": [r"강점", r"약점", r"기회", r"위협"],
    "F_PLAN_FRAMEWORK": [r"문제 ?정의", r"해결", r"실행 ?단계|Action", r"리스크|위험"],
    "F_PLAN_MINDMAP": [r"계층|가지|마인드맵"],
    "C_SUM_LENGTH_SHORT": [r"300자"],
    "C_SUM_LENGTH_LONG": [r"800자"],
    "C_SUM_NO_JARGON": [r"전문 ?용어|어려운", r"쉬운|쉽게"],
    "C_SUM_FACT_ONLY": [r"사실|팩트|객관", r"추측|주관|해석"],
    "C_SUM_ACTION_ITEM": [r"실행 ?과제|Action|조치"],
    "C_COMP_BALANCED": [r"균형|중립|편향"],
    "C_COMP_VERDICT": [r"결론|우수"],
    "C_COMP_DIFF_ONLY": [r"차이"],
    "C_COMP_COST_EFFICIENCY": [r"가성비|비용|ROI"],
    "C_COMP_SPECIFIC_METRIC": [r"기준|항목"],
    "C_EXP_EASY_ELI5": [r"쉬운|쉽게", r"비유"],
    "C_EXP_TECHNICAL_DEEP": [r"전문 ?용어|전문가|전공자"],
    "C_EXP_VISUAL_ASCII": [r"ASCII|다이어그램|흐름도"],
    "C_EXP_REAL_WORLD_CASE": [r"사례|예시"],
    "C_EXP_MISCONCEPTION": [r"오해|착각"],
    "C_REC_VALUE_FOR_MONEY": [r"가성비|가격"],
    "C_REC_STRICT_CRITERIA": [r"기준", r"근거"],
    "C_REC_USER_PERSONA": [r"상황|페르소나"],
    "C_REC_ALTERNATIVES": [r"대안|차선|대체"],
    "C_REC_CAUTION_NOTE": [r"유의|주의|단점|리스크"],
    "C_PLAN_FEASIBILITY": [r"예산|현실|실행 ?가능"],
    "C_PLAN_CREATIVE_MOONSHOT": [r"파격|독창|틀"],
    "C_PLAN_KPI_MEASURABLE": [r"KPI|지표"],
    "C_PLAN_STEP_BY_STEP": [r"단계|로드맵"],
    "C_PLAN_RISK_MITIGATION": [r"리스크|위험|문제점", r"대응|Plan B|대책"],
    "C_EXP_GENERAL": [r"용어", r"명확|정확"],
    "C_REC_GENERAL": [r"기준", r"균형|고려"],
}
assert set(KEYWORDS) == set(FORMAT_PROMPT_CATALOG) | set(CONSTRAINT_PROMPT_CATALOG)


def items_text(lb):
    return (f"역할: {lb['role']}\n목적: {lb['purpose']}\n작업: {TASK_PROMPT[lb['task']]}\n"
            f"출력 형식: {FORMAT_PROMPT_CATALOG[lb['format']]}\n제약 조건: {CONSTRAINT_PROMPT_CATALOG[lb['constraint']]}")


def rewrite_messages(lb):
    msgs = [{"role": "system", "content": REWRITE_SYSTEM}]
    for ex, out in FEW_SHOT:
        msgs += [{"role": "user", "content": items_text(ex)}, {"role": "assistant", "content": out}]
    return msgs + [{"role": "user", "content": items_text(lb)}]


NUM = re.compile(r"\d+")
FOREIGN = re.compile(r"[぀-ヿ一-鿿]")  # 3B가 가끔 섞는 한자/가나
SENT = re.compile(r"(?<=[.!?])\s+")


LIST_LINE = re.compile(r"^\s*(#|\||[-*] |\d+[.)]|\d+단계|Q[:.]|A[:.]|\*\*)")


def trim(text):
    """지시문 뒤에 이어 쓴 '답변'을 잘라낸다: 목록/제목 줄이 나오면 중단하고,
    역할 문장 이후 '~세요.'로 끝나지 않는 문장부터 버림."""
    lines = []
    for line in text.strip().splitlines():
        if LIST_LINE.match(line):
            break
        if line.strip():
            lines.append(line.strip())
    sents = SENT.split(" ".join(lines))
    keep = sents[:1]
    for s in sents[1:]:
        if not s.endswith("세요."):
            break
        keep.append(s)
    return " ".join(keep)


def josa(word, a, b):
    """받침 있으면 a, 없으면 b (예: 이/가)."""
    c = word.rstrip()[-1:] or "가"
    return a if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else b


def template_join(lb):
    """LM 없이 카탈로그 문장을 이어 붙인 자연어 지시문 (LM 실패 시 대체 경로)."""
    return (f"당신은 {lb['role']}입니다. {lb['purpose']}{josa(lb['purpose'], '이', '가')} 목적이니, "
            f"{TASK_PROMPT[lb['task']]} {FORMAT_PROMPT_CATALOG[lb['format']]} {CONSTRAINT_PROMPT_CATALOG[lb['constraint']]}")


# 반드시 보존해야 할 숫자: 분량/개수/범위(300자, 3~5개, 3문장) + 역할/목적 속 숫자(아이폰 17 등). "(1단계, 2단계...)" 같은 예시 번호는 제외
KEY_NUM = re.compile(r"\d+(?=\s*(?:자|개|문장|~))|(?<=~)\d+")


def required_numbers(lb):
    cat = FORMAT_PROMPT_CATALOG[lb["format"]] + CONSTRAINT_PROMPT_CATALOG[lb["constraint"]]
    return set(KEY_NUM.findall(cat)) | set(NUM.findall(lb["role"] + lb["purpose"]))


def verify(lb, text):
    """통과하면 빈 리스트, 아니면 실패 사유 목록."""
    errs = []
    src = items_text(lb)
    if set(FOREIGN.findall(text)) - set(FOREIGN.findall(src)):
        errs.append("외국 문자 혼입")
    if not text.startswith("당신은") or lb["role"] not in text:
        errs.append("역할 문장 없음")
    for cid in (lb["format"], lb["constraint"]):
        miss = [k for k in KEYWORDS[cid] if not re.search(k, text)]
        if miss:
            errs.append(f"{cid} 누락({', '.join(miss)})")
    if set(NUM.findall(text)) - set(NUM.findall(src)):
        errs.append(f"새 숫자 {sorted(set(NUM.findall(text)) - set(NUM.findall(src)))}")
    if required_numbers(lb) - set(NUM.findall(text)):
        errs.append(f"숫자 누락 {sorted(required_numbers(lb) - set(NUM.findall(text)))}")
    if re.search(r"^\s*(#|\||[-*] |\d+\.)", text, re.M) or "\n\n" in text.strip():
        errs.append("목록/표/여러 문단 (답변 수행 의심)")
    if not (0.6 * len(src) <= len(text) <= 1.5 * len(src)):
        errs.append(f"길이 이상 {len(text)}/{len(src)}")
    return errs


def assemble(instruction, query, content=""):
    prompt = f"{instruction} {NATURAL_PRIORITY}\n\n요청: {query}"
    if content:
        prompt += f"\n\n입력 자료:\n<<<\n{content}\n>>>"
    return prompt


class Rewriter:
    """베이스 모델로 재작성. PeftModel을 넘기면 어댑터를 끄고 같은 가중치를 공유한다."""

    def __init__(self, model=None, tok=None, base="Qwen/Qwen2.5-3B-Instruct"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = tok or AutoTokenizer.from_pretrained(base)
        self.tok.padding_side = "left"
        self.model = model or AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}).eval()

    def generate(self, labels, batch=8):
        import contextlib
        outs = []
        ctx = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()
        with ctx, self.torch.no_grad():
            for i in range(0, len(labels), batch):
                texts = [self.tok.apply_chat_template(rewrite_messages(lb), tokenize=False, add_generation_prompt=True)
                         for lb in labels[i:i + batch]]
                enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
                out = self.model.generate(**enc, max_new_tokens=300, do_sample=False, repetition_penalty=1.05)
                outs += [trim(s) for s in self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return outs

    def __call__(self, query, label, content=""):
        """-> (최종 프롬프트, style, 실패 사유). 검증 실패 시 규칙 기반 연결문으로 대체."""
        lb, _ = normalize(label, query)
        text = self.generate([lb])[0]
        errs = verify(lb, text)
        if errs:
            return assemble(template_join(lb), query, content), "template", errs
        return assemble(text, query, content), "llm", []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.preds for l in open(f, encoding="utf-8")]
    labels = [normalize(parse_label(r["pred"]), r["query"])[0] for r in rows]
    rw = Rewriter()
    t = time.time()
    texts = rw.generate(labels)
    dt = time.time() - t
    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fp:
        for r, lb, text in zip(rows, labels, texts):
            errs = verify(lb, text)
            n_ok += not errs
            fp.write(json.dumps({"query": r["query"], "label": lb, "rewrite": text, "errors": errs,
                                 "template": template_join(lb), "template_errors": verify(lb, template_join(lb)),
                                 "prompt": assemble(text if not errs else template_join(lb), r["query"])},
                                ensure_ascii=False) + "\n")
    print(f"통과 {n_ok}/{len(rows)} ({n_ok / len(rows):.1%}) | {dt:.0f}s, 건당 {dt / len(rows):.2f}s (batch 8)")
