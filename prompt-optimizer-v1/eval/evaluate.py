"""라벨 생성 평가: JSON 파싱률, task/format/constraint 정확도, 카탈로그 위반률.

평가셋
  indist  data/dataset.jsonl의 split=eval (100문항, 템플릿에서 task별 20개 층화 추출)
  ood     eval/ood.jsonl  (49문항, 템플릿 밖의 자연스러운 질문)
  ood2    eval/ood2.jsonl (40문항, 2차 학습 이후 새로 만든 공정 테스트셋)

사용 (prompt-optimizer-v1 폴더에서 실행):
  python eval/evaluate.py                            # 배포 어댑터로 세 평가셋 모두
  python eval/evaluate.py --sets ood ood2            # 일부만
  python eval/evaluate.py --adapter models/lora      # 직접 학습한 로컬 어댑터
  python eval/evaluate.py --adapter base             # 파인튜닝 전 베이스 모델
예측 결과는 eval/results/{평가셋}_preds_{이름}.jsonl에 저장된다.
"""
import argparse, collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from inference import ADAPTER, BASE_MODEL, CATALOG, SYSTEM

SETS = {
    "indist": (os.path.join(ROOT, "data", "dataset.jsonl"), lambda r: r["split"] == "eval"),
    "ood": (os.path.join(ROOT, "eval", "ood.jsonl"), None),
    "ood2": (os.path.join(ROOT, "eval", "ood2.jsonl"), None),
}


def load(name):
    path, keep = SETS[name]
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if keep is None or keep(r)]


def load_model(adapter):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
    if adapter != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.eval()


def generate(tok, model, queries, batch_size=10):
    import torch
    preds = []
    for i in range(0, len(queries), batch_size):
        msgs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}] for q in queries[i:i + batch_size]]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=160, do_sample=False)
        preds += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return preds


def score(rows, preds, outfile):
    stat = collections.Counter()
    errors = collections.Counter()
    with open(outfile, "w", encoding="utf-8") as fp:
        for r, p in zip(rows, preds):
            txt = p.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                pr = json.loads(txt)
                stat["parse"] += 1
            except Exception:
                pr = {}
            ok = {k: pr.get(k) == r[k] for k in ["task", "format", "constraint"]}
            for k, v in ok.items():
                stat[k] += v
            stat["all3"] += all(ok.values())
            t = pr.get("task")
            if t in CATALOG and (pr.get("format") not in CATALOG[t][0] or pr.get("constraint") not in CATALOG[t][1]):
                stat["catalog_violation"] += 1
            if not ok["format"]:
                errors[f"F {r['format']} -> {pr.get('format')}"] += 1
            if not ok["constraint"]:
                errors[f"C {r['constraint']} -> {pr.get('constraint')}"] += 1
            gold = {k: r[k] for k in ["role", "purpose", "task", "format", "constraint"]}
            fp.write(json.dumps({"query": r["query"], "gold": gold, "pred": p}, ensure_ascii=False) + "\n")
    return stat, errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER, help="Hugging Face 저장소 ID, 로컬 어댑터 경로, 또는 base")
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    args = ap.parse_args()

    tok, model = load_model(args.adapter)
    name = os.path.basename(os.path.normpath(args.adapter))
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    for s in args.sets:
        rows = load(s)
        preds = generate(tok, model, [r["query"] for r in rows])
        stat, errors = score(rows, preds, os.path.join(ROOT, "eval", "results", f"{s}_preds_{name}.jsonl"))
        n = len(rows)
        print(f"[{s}] n={n} " + " ".join(f"{k}={stat[k] / n:.2%}" for k in ["parse", "task", "format", "constraint", "all3", "catalog_violation"]))
        for e, c in errors.most_common(12):
            print(f"  {c:3d}  {e}")
