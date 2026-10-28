"""data/raw/dataset_*.jsonl -> chat 형식 train/eval 분리. 시스템 프롬프트에 선택 가능 카탈로그를 명시한다."""
import json, random, collections

CATALOG = {
    "summarization": (["F_SUM_BULLETS", "F_SUM_PROSE", "F_SUM_KEYWORD", "F_SUM_TIMELINE"],
                      ["C_SUM_LENGTH_SHORT", "C_SUM_LENGTH_LONG", "C_SUM_NO_JARGON", "C_SUM_FACT_ONLY", "C_SUM_ACTION_ITEM"]),
    "comparison": (["F_COMP_TABLE", "F_COMP_PRO_CON", "F_COMP_VENN"],
                   ["C_COMP_BALANCED", "C_COMP_VERDICT", "C_COMP_DIFF_ONLY", "C_COMP_COST_EFFICIENCY", "C_COMP_SPECIFIC_METRIC"]),
    "explanation": (["F_EXP_STEPWISE", "F_EXP_QA", "F_EXP_DEEPDIVE"],
                    ["C_EXP_EASY_ELI5", "C_EXP_TECHNICAL_DEEP", "C_EXP_VISUAL_ASCII", "C_EXP_REAL_WORLD_CASE", "C_EXP_MISCONCEPTION"]),
    "recommendation": (["F_REC_TOPN", "F_REC_SITUATIONAL", "F_REC_PRO_CON", "F_REC_SCORECARD"],
                       ["C_REC_VALUE_FOR_MONEY", "C_REC_STRICT_CRITERIA", "C_REC_USER_PERSONA", "C_REC_ALTERNATIVES", "C_REC_CAUTION_NOTE"]),
    "planning": (["F_PLAN_IDEA_LIST", "F_PLAN_SWOT", "F_PLAN_FRAMEWORK", "F_PLAN_MINDMAP"],
                 ["C_PLAN_FEASIBILITY", "C_PLAN_CREATIVE_MOONSHOT", "C_PLAN_KPI_MEASURABLE", "C_PLAN_STEP_BY_STEP", "C_PLAN_RISK_MITIGATION"]),
}


def build_system_prompt():
    lines = [
        "사용자 질문을 분석해 AI의 role, purpose, task, format, constraint를 JSON 한 줄로 출력하세요.",
        "",
        "[선택 가능 카탈로그]",
        "task: " + ", ".join(CATALOG),
        "",
        "format (task별, 첫 번째가 디폴트):",
    ]
    for t, (f, _) in CATALOG.items():
        lines.append(f"- {t}: " + ", ".join(f))
    lines += ["", "constraint (task별, 첫 번째가 디폴트):"]
    for t, (_, c) in CATALOG.items():
        lines.append(f"- {t}: " + ", ".join(c))
    lines += [
        "",
        "[규칙]",
        "- format과 constraint는 반드시 선택한 task의 목록 안에서만 고르세요.",
        "- 질문에 형식이나 조건이 명시되지 않으면 해당 task의 디폴트를 고르세요.",
        '- 출력 키 순서: {"role", "purpose", "task", "format", "constraint"}',
    ]
    return "\n".join(lines)


SYSTEM = build_system_prompt()
KEYS = ["role", "purpose", "task", "format", "constraint"]


def to_chat(r):
    answer = json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["query"]},
        {"role": "assistant", "content": answer},
    ], "label": {k: r[k] for k in KEYS}}


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/dataset_v2.jsonl"
    sfx = sys.argv[2] if len(sys.argv) > 2 else "_v2"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    rng = random.Random(0)
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    train, evals = [], []
    for t, rs in by_task.items():  # task별 층화 분리 (90/10)
        rng.shuffle(rs)
        evals += rs[:20]
        train += rs[20:]
    rng.shuffle(train)
    for name, data in [(f"data/train/train{sfx}.jsonl", train), (f"data/train/eval{sfx}.jsonl", evals)]:
        with open(name, "w", encoding="utf-8") as fp:
            for r in data:
                fp.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
    print("train", len(train), "eval", len(evals))
