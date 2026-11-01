"""Qwen2.5-3B-Instruct LoRA SFT. assistant(JSON) 토큰에만 loss를 건다.

데이터: data/dataset.jsonl (한 줄에 질문 하나와 정답 라벨)
  source  template = 템플릿으로 생성한 질문, natural = 직접 작성한 자연어 질문
  split   train / eval (eval은 template에서 task별 20개씩 층화 추출)
학습셋 = template train + natural x NATURAL_OVERSAMPLE, 검증셋 = eval

사용: python train.py [출력 폴더] [lr] [epochs]     예) python train.py models/lora 2e-4 2
"""
import json, random, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from inference import BASE_MODEL, KEYS, SYSTEM
# KEYS = ["role", "purpose", "task", "format", "constraint"]
# SYSTEM = build_system_prompt()

DATA = "data/dataset.jsonl"
# python train.py models/my_model 0.001 2e-3 3 과 같이 하이퍼파라미터를 터미널에서 바꾸어서 실행하고 싶을 때 sys.argv를 사용한다.
# 대부분의 경우 사용하지 않으므로 기본값을 보면 된다
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/lora"
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4
EPOCHS = float(sys.argv[3]) if len(sys.argv) > 3 else 2
# v1의 데이터세트는 정해진 규격에 맞추어 rule-based로 생성해낸 템플릿 데이터 900개와, 모델이 별도의 추론능력을 사용하여 생성한 데이터 300개가 있다.
# 후자의 데이터 품질이 높으므로 후자의 경우 비중을 키워서 학습한다.
NATURAL_OVERSAMPLE = 2  # 자연어 320개 vs 템플릿 900개 -> 자연어 비중을 키움

tok = AutoTokenizer.from_pretrained(BASE_MODEL)


def to_messages(r): # r은 {query, role, purpose, task, format, constraint, split, source}로 구성되어있음
    # 답지 생성 {role:, purpose:, task:, format:, constraint:} 하여 문자열로 반환한다.
    answer = json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False)
    # 생성한 답지와 템플릿을 결합한다.
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["query"]},
            {"role": "assistant", "content": answer}]


def encode(r):
    msgs = to_messages(r)
    # 앞서 role, content 형식의 json을 챗 탬플릿으로 반환한다. (<|im_start|>user assistant <|im_end|>)
    # role:assistant, contet:answer 이전까지 (요청)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    # role:assistant, contet:answer 부분 (답변)
    full = prompt + msgs[-1]["content"] + tok.eos_token  # <|im_end|>
    # 요청 부분 토큰화
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    # 대답 부분 토큰화
    ids = tok(full, add_special_tokens=False)["input_ids"]
    # 요청 부분은 로스에서 제외하므로 마스크(-100)처리
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "labels": labels}


def load_splits(path):
    # jsonl 데이터를 한 줄 씩 열어서 딕셔너리 데이터로 변환(json.loads)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    # split이 train인 템플릿 데이터를 학습 데이터로 넣음
    train = [r for r in rows if r["split"] == "train" and r["source"] == "template"]
    # 자연어인 데이터들은 2배 뻥튀기해서 넣음
    train += [r for r in rows if r["split"] == "train" and r["source"] == "natural"] * NATURAL_OVERSAMPLE
    # 섞어줌
    random.Random(1).shuffle(train)
    evals = [r for r in rows if r["split"] == "eval"]
    return train, evals


class DS(torch.utils.data.Dataset): 
    # 앞서 리스트로 받은 데이터들을 파이토치에서 사용하는 데이터 세트로 변환함
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def collate(batch): # 배치단위 데이터콜레이터
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    # 배치에서 가장 길이가 긴 데이터 기준으로 패딩
    # 요청
    ids = torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch])
    # 정답
    lab = torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch])
    # 어텐션 마스크 (패딩은 0, 패딩 아닌 토큰은 1)
    att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


if __name__ == "__main__":
    # jsonl
    train_rows, eval_rows = load_splits(DATA)
    # tensor (토큰화)
    train, evals = [encode(r) for r in train_rows], [encode(r) for r in eval_rows]
    lens = [len(r["input_ids"]) for r in train]
    print(f"train {len(train)} eval {len(evals)} | seq len min {min(lens)} max {max(lens)} | "
          f"answer tokens avg {sum(sum(x != -100 for x in r['labels']) for r in train) / len(train):.1f}")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    # 역전파시 계산 그래프의 모든 지점을 기억하지 않고 체크포인팅 (연산 그래프 자체를 생성하지 않음)
    model.gradient_checkpointing_enable()
    # LORA를 사용하기에 모델을 얼려두고 모델 자체의 미분은 가능하도록 함
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
        # 학습 성적표 기록
        json.dump(trainer.state.log_history, fp, indent=1)
