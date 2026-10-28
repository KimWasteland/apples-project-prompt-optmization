"""natural_raw.txt(직접 작성한 자연어 질문) -> natural.jsonl, 그리고 v3 학습셋 = 템플릿 v2 train + 자연어 x OVERSAMPLE."""
import json, random, collections
from prep import CATALOG, to_chat

TASKS = {"S": "summarization", "C": "comparison", "E": "explanation", "R": "recommendation", "P": "planning"}
PREFIX = {"summarization": "SUM", "comparison": "COMP", "explanation": "EXP", "recommendation": "REC", "planning": "PLAN"}
OVERSAMPLE = 2  # 자연어 320개 vs 템플릿 900개 -> 자연어 비중을 키움

rows = []
for line in open("data/raw/natural_raw.txt", encoding="utf-8"):
    if not line.strip() or line.startswith("#"):
        continue
    q, t, f, c, role, purpose = [x.strip() for x in line.split("|")]
    task = TASKS[t]
    p = PREFIX[task]
    if task == "planning":
        purpose += " 기획 아이디어 도출"
    r = {"query": q, "role": role, "purpose": purpose, "task": task, "format": f"F_{p}_{f}", "constraint": f"C_{p}_{c}"}
    fids, cids = CATALOG[task]
    assert r["format"] in fids and r["constraint"] in cids, r
    rows.append(r)

assert len({r["query"] for r in rows}) == len(rows), "중복 질문"
tests = {json.loads(l)["messages"][1]["content"] for f in ["data/test/ood.jsonl", "data/test/ood2.jsonl"] for l in open(f, encoding="utf-8")}
assert not tests & {r["query"] for r in rows}, "테스트셋과 겹침"

with open("data/raw/natural.jsonl", "w", encoding="utf-8") as fp:
    for r in rows:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

train = [json.loads(l) for l in open("data/train/train_v2.jsonl", encoding="utf-8")]
train += [to_chat(r) for r in rows] * OVERSAMPLE
random.Random(1).shuffle(train)
with open("data/train/train_v3.jsonl", "w", encoding="utf-8") as fp:
    for r in train:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

print("natural", len(rows), collections.Counter(r["task"] for r in rows))
print("default F/C:", sum(r["format"] == CATALOG[r["task"]][0][0] for r in rows), sum(r["constraint"] == CATALOG[r["task"]][1][0] for r in rows))
print("train_v3", len(train))
