"""생성 평가: JSON 파싱률, task/format/constraint 정확도, 카탈로그 위반률."""
import json, sys, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prep import CATALOG

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "base" else None
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "results/preds/eval_preds.jsonl"
DATA = sys.argv[3] if len(sys.argv) > 3 else "data/train/eval_v3.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
preds = []
for i in range(0, len(rows), 10):
    batch = rows[i:i + 10]
    prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=160, do_sample=False)
    preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

stat = collections.Counter()
errors = collections.Counter()
with open(OUTFILE, "w", encoding="utf-8") as fp:
    for r, p in zip(rows, preds):
        gold = r["label"]
        txt = p.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            pr = json.loads(txt)
            stat["parse"] += 1
        except Exception:
            pr = {}
        ok = {k: pr.get(k) == gold[k] for k in ["task", "format", "constraint"]}
        for k, v in ok.items():
            stat[k] += v
        stat["all3"] += all(ok.values())
        t = pr.get("task")
        if t in CATALOG and (pr.get("format") not in CATALOG[t][0] or pr.get("constraint") not in CATALOG[t][1]):
            stat["catalog_violation"] += 1
        if not ok["format"]:
            errors[f"F {gold['format']} -> {pr.get('format')}"] += 1
        if not ok["constraint"]:
            errors[f"C {gold['constraint']} -> {pr.get('constraint')}"] += 1
        fp.write(json.dumps({"query": r["messages"][1]["content"], "gold": gold, "pred": p}, ensure_ascii=False) + "\n")

n = len(rows)
print(f"[{ADAPTER or 'base'}] n={n} " + " ".join(f"{k}={stat[k] / n:.2%}" for k in ["parse", "task", "format", "constraint", "all3", "catalog_violation"]))
print("top errors:")
for e, c in errors.most_common(12):
    print(f"  {c:3d}  {e}")
