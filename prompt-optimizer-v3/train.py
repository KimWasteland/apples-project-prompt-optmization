"""v3 학습. Qwen2.5-3B-Instruct QLoRA (Unsloth + TRL). v3, v4는 같은 설정이다.

사용:
  python prompt-optimizer-v3/train.py

결과
  prompt-optimizer-v3/lora_adapter/     학습된 어댑터
  prompt-optimizer-v3/test_samples.txt  test 3문항 생성 결과
  prompt-optimizer-v3/checkpoints/      중간 저장 (학습 후 지워도 된다)
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 12GB GPU 메모리 부족 완화

from unsloth import FastLanguageModel   # unsloth는 transformers, trl보다 먼저 불러와야 한다
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from optimizer import BASE_MODEL, build_messages
from utils import DATA, HERE, read_jsonl

MAX_LENGTH = 2048
SEED = 42


def to_messages(row):
    """학습 데이터 한 행 -> 대화 (system, user, assistant)"""
    messages = build_messages(row["input"])
    messages.append({"role": "assistant", "content": row["output"]})
    return messages


def to_dataset(rows, split, tokenizer):
    texts = []
    for row in rows:
        if row["split"] == split:
            texts.append({"text": tokenizer.apply_chat_template(to_messages(row), tokenize=False)})
    return Dataset.from_list(texts)


def write_test_samples(model, tokenizer, rows, path):
    """test 앞 3문항을 생성해서 눈으로 확인할 수 있게 저장한다."""
    FastLanguageModel.for_inference(model)
    test_rows = [row for row in rows if row["split"] == "test"][:3]
    with open(path, "w", encoding="utf-8") as f:
        for row in test_rows:
            messages = to_messages(row)[:-1]   # 정답(assistant)은 빼고 넣는다
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True).to(model.device)
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            input_length = inputs["input_ids"].shape[1]
            prediction = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
            f.write(f"### INPUT\n{row['input']}\n\n### PRED\n{prediction}\n\n### GOLD\n{row['output']}\n\n{'=' * 60}\n\n")


def main():
    folder = HERE
    rows = read_jsonl(DATA / "dataset_v3.jsonl")

    # 4bit로 올린 베이스 모델에 LoRA를 붙인다
    model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=MAX_LENGTH, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=SEED,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")

    settings = SFTConfig(
        output_dir=str(folder / "checkpoints"),
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,   # 2 x 8 = 유효 배치 16
        learning_rate=2e-4,
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
        seed=SEED,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=settings,
                         train_dataset=to_dataset(rows, "train", tokenizer),
                         eval_dataset=to_dataset(rows, "val", tokenizer))
    # 정답(assistant) 부분에서만 loss를 계산한다
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    trainer.train()

    model.save_pretrained(str(folder / "lora_adapter"))
    tokenizer.save_pretrained(str(folder / "lora_adapter"))
    write_test_samples(model, tokenizer, rows, folder / "test_samples.txt")


if __name__ == "__main__":
    main()
