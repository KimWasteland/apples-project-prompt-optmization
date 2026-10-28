"""Qwen2.5-3B-Instruct LoRA SFT. assistant(JSON) 토큰에만 loss를 건다."""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/lora"
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4
EPOCHS = float(sys.argv[3]) if len(sys.argv) > 3 else 3

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + msgs[-1]["content"] + tok.eos_token  # <|im_end|>
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels}


def load(path):
    return [encode(json.loads(l)) for l in open(path, encoding="utf-8")]


class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


if __name__ == "__main__":
    sfx = sys.argv[4] if len(sys.argv) > 4 else "_v3"
    train, evals = load(f"data/train/train{sfx}.jsonl"), load(f"data/train/eval{sfx}.jsonl")
    lens = [len(r["input_ids"]) for r in train]
    print(f"seq len min {min(lens)} max {max(lens)} | answer tokens avg {sum(sum(x != -100 for x in r['labels']) for r in train) / len(train):.1f}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        # 12GB VRAM(WDDM): 한계 근처에서 공유 메모리로 넘쳐 step당 15~27초로 느려짐 -> batch 1 x accum 16
        output_dir=OUT, per_device_train_batch_size=1, per_device_eval_batch_size=2,
        gradient_accumulation_steps=16, learning_rate=LR, num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine", warmup_steps=8, weight_decay=0.0,
        bf16=True, logging_steps=5, eval_strategy="steps", eval_steps=20,
        save_strategy="no", report_to="none", dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(train), eval_dataset=DS(evals), data_collator=collate)
    trainer.train()
    model.save_pretrained(OUT)
    with open(f"{OUT}/log_history.json", "w") as fp:
        json.dump(trainer.state.log_history, fp, indent=1)
