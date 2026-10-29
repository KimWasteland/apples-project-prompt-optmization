"""data/raw/*.txt 파싱 → 검증 → master.jsonl / SFT 분할 파일 생성.

사용법:
    python scripts/build_dataset.py            # 파싱 + 검증 + 분포 리포트
    python scripts/build_dataset.py --export   # 검증 통과 시 SFT 파일까지 생성
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FINAL_DIR = ROOT / "data" / "final"
CONFIG = json.loads((ROOT / "config" / "quota.json").read_text(encoding="utf-8"))

LEVELS = {"L1", "L2", "L3"}
SECTIONS = ["[과제 유형]", "[역할]", "[과제]", "[맥락 및 가정]", "[원문]", "[수행 단계]", "[출력 형식]"]
OPTIONAL_SECTIONS = {"[원문]"}
FORMAT_KEYS = ["- 형식:", "- 분량:", "- 톤:", "- 반드시 포함:"]
BAD_PLACEHOLDERS = ["OOO", "○○○", "XXX"]


def parse_file(path):
    samples = []
    blocks = re.split(r"^@@@ ", path.read_text(encoding="utf-8"), flags=re.M)
    for block in blocks[1:]:
        header, _, body = block.partition("\n")
        fields = [f.strip() for f in header.split("|")]
        m = re.match(r"\s*>>> INPUT\s*\n(.*?)\n>>> OUTPUT\s*\n(.*)", body, flags=re.S)
        sample = {"file": path.name, "header": fields}
        if len(fields) == 6:
            sample.update(zip(["id", "task", "level", "domain", "persona", "tone"], fields))
        if m:
            sample["input"] = m.group(1).strip()
            sample["output"] = m.group(2).strip()
        samples.append(sample)
    return samples


def split_sections(output):
    """섹션 헤더 → 본문. 헤더는 줄 시작에만 인정."""
    found = []
    for name in SECTIONS:
        for m in re.finditer(r"^" + re.escape(name), output, flags=re.M):
            found.append((m.start(), name))
    found.sort()
    sections = {}
    for i, (pos, name) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(output)
        sections[name] = output[pos + len(name):end].strip()
    return [name for _, name in found], sections


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def validate(s):
    errors = []
    if len(s["header"]) != 6:
        return [f"헤더 필드 수 {len(s['header'])}개 (6개여야 함)"]
    if "input" not in s:
        return ["INPUT/OUTPUT 블록 없음"]
    if s["task"] not in CONFIG["tasks"]:
        errors.append(f"task 값 오류: {s['task']}")
    if s["level"] not in LEVELS:
        errors.append(f"level 값 오류: {s['level']}")
    if s["domain"] not in CONFIG["domains"]:
        errors.append(f"domain 값 오류: {s['domain']}")
    if s["persona"] not in CONFIG["personas"]:
        errors.append(f"persona 값 오류: {s['persona']}")
    if s["tone"] not in CONFIG["tones"]:
        errors.append(f"tone 값 오류: {s['tone']}")
    prefix = CONFIG["id_prefix"].get(s["task"])
    if prefix and not s["id"].startswith(prefix + "_"):
        errors.append(f"id 접두사가 task와 다름: {s['id']}")

    order, sections = split_sections(s["output"])
    required = [n for n in SECTIONS if n not in OPTIONAL_SECTIONS]
    missing = [n for n in required if n not in order]
    if missing:
        errors.append(f"섹션 누락: {', '.join(missing)}")
    if len(order) != len(set(order)):
        errors.append("섹션 중복")
    if order != [n for n in SECTIONS if n in order]:
        errors.append(f"섹션 순서 오류: {' '.join(order)}")
    if not s["output"].startswith("[과제 유형]"):
        errors.append("출력이 [과제 유형]으로 시작하지 않음")
    if sections.get("[과제 유형]") != s["task"]:
        errors.append(f"[과제 유형] 값({sections.get('[과제 유형]')})이 헤더 task와 다름")
    fmt = sections.get("[출력 형식]", "")
    for key in FORMAT_KEYS:
        if key not in fmt:
            errors.append(f"[출력 형식]에 '{key}' 없음")
    if "[원문]" in sections and norm(sections["[원문]"]) not in norm(s["input"]):
        errors.append("[원문] 내용이 입력에 그대로 존재하지 않음")
    if not re.search(r"^1\. ", sections.get("[수행 단계]", ""), flags=re.M):
        errors.append("[수행 단계]가 번호 목록이 아님")
    for bad in BAD_PLACEHOLDERS:
        if bad in s["output"]:
            errors.append(f"잘못된 빈칸 표기: {bad} ({{{{...}}}} 사용)")
    for name, body in sections.items():
        if not body:
            errors.append(f"{name} 내용이 비어 있음")
    return errors


def report(samples):
    print(f"\n총 {len(samples)}개 / 목표 {CONFIG['total']}개")
    by_cell = Counter((s["task"], s["level"]) for s in samples)
    print("\n[task × level] 현재/목표")
    for task, levels in CONFIG["tasks"].items():
        cells = "  ".join(f"{lv} {by_cell[(task, lv)]:>3}/{n:<3}" for lv, n in levels.items())
        print(f"  {task}: {cells}")
    for key in ["domain", "persona", "tone"]:
        c = Counter(s[key] for s in samples)
        print(f"\n[{key}] " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    roles = Counter()
    for s in samples:
        _, sec = split_sections(s["output"])
        role = sec.get("[역할]", "")
        roles[re.sub(r"\s+", " ", role)[:20]] += 1
    common = [(r, n) for r, n in roles.most_common(5) if n > 1]
    if common:
        print("\n[역할 첫 20자 반복 상위]")
        for r, n in common:
            print(f"  {n:>3}  {r}")
    steps = Counter(
        len(re.findall(r"^\d+\. ", split_sections(s["output"])[1].get("[수행 단계]", ""), flags=re.M))
        for s in samples
    )
    print("\n[수행 단계 수] " + ", ".join(f"{k}단계 {v}" for k, v in sorted(steps.items())))
    lens = sorted(len(s["output"]) for s in samples)
    if lens:
        print(f"\n[출력 글자 수] 최소 {lens[0]} / 중앙 {lens[len(lens) // 2]} / 최대 {lens[-1]}")


def to_messages(s):
    return {
        "messages": [
            {"role": "system", "content": CONFIG["system_prompt"]},
            {"role": "user", "content": f"<prompt>\n{s['input']}\n</prompt>"},
            {"role": "assistant", "content": s["output"]},
        ]
    }


def export(samples):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    meta_keys = ["id", "task", "level", "domain", "persona", "tone", "input", "output"]
    with open(FINAL_DIR / "master.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({k: s[k] for k in meta_keys}, ensure_ascii=False) + "\n")

    cfg = CONFIG["split"]
    rng = random.Random(cfg["seed"])
    groups = defaultdict(list)
    for s in samples:
        groups[(s["task"], s["level"])].append(s)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        n_val = round(len(items) * cfg["val"])
        n_test = round(len(items) * cfg["test"])
        splits["val"] += items[:n_val]
        splits["test"] += items[n_val:n_val + n_test]
        splits["train"] += items[n_val + n_test:]
    for name, items in splits.items():
        rng.shuffle(items)
        with open(FINAL_DIR / f"sft_{name}.jsonl", "w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps(to_messages(s), ensure_ascii=False) + "\n")
        with open(FINAL_DIR / f"ids_{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(s["id"] for s in items) + "\n")
        print(f"  sft_{name}.jsonl: {len(items)}개")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="분포 리포트 생략")
    args = ap.parse_args()

    samples = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        samples += parse_file(path)

    n_err = 0
    for s in samples:
        for e in validate(s):
            n_err += 1
            print(f"[오류] {s['file']} {s.get('id', '?')}: {e}")
    for key, label in [("id", "id"), ("input", "입력")]:
        seen = {}
        for s in samples:
            k = norm(s.get(key, "")).lower()
            if k in seen:
                n_err += 1
                print(f"[오류] {label} 중복: {s.get('id')} ↔ {seen[k]}")
            seen[k] = s.get("id")

    valid = [s for s in samples if len(s["header"]) == 6 and "input" in s]
    if not args.quiet:
        report(valid)
    print(f"\n오류 {n_err}건")
    if args.export:
        if n_err:
            print("오류가 있어 export하지 않음")
            sys.exit(1)
        export(valid)


if __name__ == "__main__":
    main()
