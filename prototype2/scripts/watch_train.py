"""학습 로그 감시: 이상 징후나 학습 종료 시 메시지를 출력하고 종료한다.

종료 조건
- NaN/inf loss, Traceback, CUDA OOM
- train loss 급등(직전 최저값의 1.5배 초과, 20 step 이후)
- eval_loss 2회 연속 상승(과적합 신호)
- 학습 완료([샘플 생성] 출력)
"""
import re
import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "outputs" / "train_log.txt"
POLL_SEC = 20


def parse(text):
    train, evals = [], []
    for m in re.finditer(r"\{[^{}]*\}", text):
        d = dict(re.findall(r"'(\w+)': '?([^',}]+)'?", m.group(0)))
        if "loss" in d and "grad_norm" in d:
            train.append((float(d.get("epoch", 0)), d["loss"], d.get("grad_norm")))
        elif "eval_loss" in d:
            evals.append((float(d.get("epoch", 0)), d["eval_loss"]))
    return train, evals


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    while True:
        text = LOG.read_text(encoding="utf-8", errors="ignore") if LOG.exists() else ""
        if "Traceback" in text or "out of memory" in text.lower():
            idx = max(text.rfind("Traceback"), text.lower().rfind("out of memory"))
            print("[이상] 학습 중 오류 발생\n" + text[idx:idx + 2000])
            return
        train, evals = parse(text)
        for ep, loss, gn in train:
            if loss.lower() in ("nan", "inf") or (gn and gn.lower() in ("nan", "inf")):
                print(f"[이상] NaN/inf 발생 epoch={ep} loss={loss} grad_norm={gn}")
                return
        losses = [float(l) for _, l, _ in train]
        if len(losses) >= 4:
            best = min(losses[:-1])
            if losses[-1] > best * 1.5:
                print(f"[이상] train loss 급등: 최저 {best:.3f} → 최근 {losses[-1]:.3f}\n최근 로그: {train[-5:]}")
                return
        ev = [float(e) for _, e in evals]
        if len(ev) >= 3 and ev[-1] > ev[-2] > ev[-3]:
            print(f"[이상] eval_loss 2회 연속 상승(과적합 신호): {evals[-4:]}")
            return
        if "[샘플 생성]" in text:
            print(f"[완료] train loss 추이: {[l for _, l, _ in train]}\neval loss 추이: {evals}")
            return
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
