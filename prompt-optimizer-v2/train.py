"""Qwen2.5-3B-Instruct 프롬프트 최적화 QLoRA SFT (Unsloth + TRL).

데이터: data/dataset.jsonl (split: train / val / test, task × level별 층화 분할)

사용법:
    python train.py --max_steps 5      # 스모크 테스트
    python train.py                    # 본 학습 (기본 3 epoch)
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 12GB VRAM 단편화 완화

import unsloth  # noqa: F401  # transformers/trl보다 먼저 import해야 패치가 적용됨
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from inference import build_messages

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--output_dir", default=str(ROOT / "outputs" / "qwen2.5-3b-promptopt"))
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--resume", default=None, help="이어서 학습할 체크포인트 경로")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_split(name):
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    return [r for r in rows if r["split"] == name]


def to_messages(r):
    return build_messages(r["input"]) + [{"role": "assistant", "content": r["output"]}]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def to_text(split):
        return Dataset.from_list([{"text": tokenizer.apply_chat_template(to_messages(r), tokenize=False)}
                                  for r in load_split(split)])

    train_ds, val_ds = to_text("train"), to_text("val")

    lengths = [len(tokenizer(t)["input_ids"]) for t in train_ds["text"]]
    print(f"[데이터] train {len(train_ds)} / val {len(val_ds)}")
    print(f"[토큰 길이] train 최대 {max(lengths)} / 평균 {sum(lengths) / len(lengths):.0f}")
    too_long = sum(n > args.max_seq_length for n in lengths)
    if too_long:
        print(f"[경고] max_seq_length 초과 {too_long}개는 잘림")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=args.max_seq_length,
            packing=False,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="epoch",
            save_total_limit=3,
            seed=args.seed,
            report_to="none",
            dataloader_num_workers=0,
        ),
    )
    # system/user 토큰은 loss에서 제외, assistant 응답만 학습
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train(resume_from_checkpoint=args.resume)
    print(stats)

    adapter_dir = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[저장] {adapter_dir}")

    # 테스트셋 앞 3개로 생성 결과 확인
    FastLanguageModel.for_inference(model)
    out_path = Path(args.output_dir) / "test_samples.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in load_split("test")[:3]:
            inputs = tokenizer.apply_chat_template(
                build_messages(r["input"]), add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)
            gen = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            pred = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(f"### INPUT\n{r['input']}\n\n### PRED\n{pred}\n\n### GOLD\n{r['output']}\n\n{'=' * 60}\n\n")
    print(f"[샘플 생성] {out_path}")


if __name__ == "__main__":
    main()
