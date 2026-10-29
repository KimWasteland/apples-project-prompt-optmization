# 프롬프트 최적화기 (Qwen2.5-3B + QLoRA)

대충 쓴 프롬프트를 **역할 명시 · 모호한 지시 명확화 · 구조화 · 출력 형식 명시화**를 적용한 프롬프트로 바꿔 주는 모델입니다.
Qwen2.5-3B-Instruct를 Unsloth 4-bit QLoRA로 SFT 했습니다. 대상 task는 설명 · 요약 · 기획 · 비교 · 추천(+기타)입니다.

```
입력:  블록체인 설명해줘

출력:  [과제 유형] 설명
       [역할] 당신은 ... 
       [과제] ...
       [맥락 및 가정] - 가정: ... / {{대상 독자}}
       [수행 단계] 1. ... 2. ... 3. ...
       [출력 형식] - 형식: / - 분량: / - 톤: / - 반드시 포함:
```

## 구성

```
app/                 데모 웹 서버 (FastAPI + 스트리밍, 파인튜닝 전/후 비교)
config/quota.json    task·level별 데이터 개수, 도메인·페르소나·말투 목록, system 프롬프트
data/raw/            학습 데이터 원본 (사람이 읽는 txt 형식)
data/final/          export된 SFT 데이터 (messages JSONL, train/val/test)
data/extra/          학습에 쓰지 않은 예비 데이터 172개 (평가용으로 활용 가능)
docs/                데이터 작성 가이드, 학습 직후 테스트 샘플 출력
outputs/.../lora_adapter/   학습된 LoRA 어댑터 (바로 로드 가능)
scripts/build_dataset.py    원본 파싱 · 검증 · export
scripts/train_sft.py        QLoRA SFT 학습
scripts/watch_train.py      학습 로그 감시 (NaN, loss 급등, 과적합 감지)
```

## 설치

NVIDIA GPU(VRAM 12GB 이상 권장)가 필요합니다. 검증 환경은 Python 3.11, Windows 11, RTX 5070 12GB, CUDA 12.8입니다.

```bash
git lfs install                      # 어댑터 가중치가 LFS로 저장되어 있음
git clone <이 저장소 주소>
cd <저장소 폴더>

conda create -n promptopt python=3.11 -y
conda activate promptopt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> clone 후 `outputs/qwen2.5-3b-promptopt/lora_adapter/adapter_model.safetensors`가 1KB 정도라면 LFS 파일이 내려받아지지 않은 것입니다. `git lfs pull`을 실행하세요.

베이스 모델(`unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`, `Qwen/Qwen2.5-3B-Instruct`)은 처음 실행할 때 Hugging Face에서 자동으로 내려받습니다. API 키는 필요 없습니다.

## 1. 학습된 모델 써 보기 (데모)

```bash
python app/server.py
```

브라우저에서 http://127.0.0.1:7860 을 열면 됩니다.

- 테스트셋 예시를 불러오거나 직접 프롬프트를 입력하고 **최적화**를 누릅니다 (Ctrl+Enter).
- **파인튜닝 전과 비교**를 누르면 같은 입력을 LoRA를 끈 base 모델에도 넣어 나란히 보여 줍니다.
- 출력 아래 배지로 템플릿 준수 여부(섹션 6개, 순서, 출력 형식 4개 키, 빈칸 수)를 바로 확인할 수 있습니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `ADAPTER_DIR` | `outputs/qwen2.5-3b-promptopt/lora_adapter` | 로드할 어댑터 경로 (직접 학습한 체크포인트로 교체 가능) |
| `HOST` | `127.0.0.1` | 같은 네트워크의 다른 기기에서 접속하려면 `0.0.0.0` |
| `PORT` | `7860` | 서버 포트 |

외부에 공개하려면 ngrok 같은 터널을 쓰면 됩니다: `ngrok http 7860`

### 코드에서 직접 로드

```python
from unsloth import FastLanguageModel
import json

model, tokenizer = FastLanguageModel.from_pretrained(
    "outputs/qwen2.5-3b-promptopt/lora_adapter", max_seq_length=2048, load_in_4bit=True
)
FastLanguageModel.for_inference(model)

system = json.load(open("config/quota.json", encoding="utf-8"))["system_prompt"]
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "<prompt>\n블록체인 설명해줘\n</prompt>"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

입력은 반드시 학습 때와 같은 system 프롬프트와 `<prompt>...</prompt>` 감싸기를 지켜야 합니다.

## 2. 직접 학습하기

### 데이터 검증 · export

```bash
python scripts/build_dataset.py            # 검증 + 분포 리포트
python scripts/build_dataset.py --export   # 오류 0개일 때 data/final/ 생성 (task×level 층화 9:0.5:0.5)
```

`data/final/`은 이미 포함되어 있으므로, 원본을 고치지 않았다면 이 단계는 건너뛰어도 됩니다.

### 학습

```bash
python scripts/train_sft.py --max_steps 5   # 스모크 테스트 (몇 분)
python scripts/train_sft.py                 # 본 학습: 3 epoch, 174 step
```

다른 터미널에서 로그를 감시하려면 학습 출력을 파일로 남기고 감시 스크립트를 실행합니다.

```bash
python scripts/train_sft.py > outputs/train_log.txt 2>&1
python scripts/watch_train.py
```

중간에 멈췄다면 체크포인트에서 이어서 학습할 수 있습니다.

```bash
python scripts/train_sft.py --resume outputs/qwen2.5-3b-promptopt/checkpoint-58
```

| 설정 | 값 |
|---|---|
| 베이스 | Qwen2.5-3B-Instruct, 4-bit (Unsloth) |
| LoRA | r=16, alpha=32, dropout 0, q/k/v/o/gate/up/down |
| 배치 | batch 2 × grad_accum 8 = 유효 16 |
| 학습률 | 2e-4, cosine, warmup 3% |
| 기타 | max_seq_length 2048, adamw_8bit, bf16, assistant 응답만 loss 계산 |

학습이 끝나면 어댑터가 `outputs/qwen2.5-3b-promptopt/lora_adapter`에 저장되고(포함된 어댑터를 덮어씀), 테스트셋 3개의 생성 결과가 `test_samples.txt`로 남습니다.

**VRAM이 부족하면** `--batch_size 1 --grad_accum 16`으로 줄이세요. 12GB에서 batch 4는 OOM이 났습니다.

## 데이터셋

- 총 1,031개 (train 923 / val 54 / test 54), 검증 오류 0
- task × level

| task | L1 | L2 | L3 | 합계 |
|---|---|---|---|---|
| 설명 | 78 | 100 | 30 | 208 |
| 요약 | 78 | 95 | 30 | 203 |
| 기획 | 65 | 95 | 30 | 190 |
| 비교 | 65 | 95 | 30 | 190 |
| 추천 | 65 | 95 | 30 | 190 |
| 기타 | 17 | 25 | 8 | 50 |

- **L1** 키워드 수준(정보 거의 없음) / **L2** 일부 정보 있음, 형식·범위 불분명 / **L3** 이미 구체적, 원본 조건을 전부 보존하고 구조만 보완
- 도메인 12개: IT·개발, 교육·학습, 비즈니스·마케팅, 금융·경제, 건강·의료, 여행, 요리·생활, 학술·연구, 법·행정, 문화·엔터, 커리어, 육아·관계
- 페르소나 10개: 학생, 직장인, 개발자, 자영업자, 교사, 연구자, 주부, 시니어, 취준생, 프리랜서
- 말투: 존댓말 41% · 반말 29% · 영어혼용 15% · 구어체(오타·줄임말) 15%

원본 형식과 작성 규칙은 [docs/authoring_guide.md](docs/authoring_guide.md), [docs/agent_brief.md](docs/agent_brief.md)를 참고하세요.

```
@@@ exp_0002 | 설명 | L1 | 금융·경제 | 주부 | 존댓말
>>> INPUT
인플레이션이 뭔가요?
>>> OUTPUT
[과제 유형] 설명
...
```

## 학습 결과

- train loss 2.10 → 약 0.76, eval loss 1.022 → 0.958 (3 epoch 동안 계속 감소, 과적합 징후 없음)
- 테스트 샘플에서 템플릿 준수, 질문에 직접 답해 버리는 문제 없음 ([docs/test_samples.txt](docs/test_samples.txt))

### 알려진 한계

- 입력에 없는 요구 사항을 지어내는 경우가 있음
- 역할 문구가 과제와 어긋나는 경우가 있음 (예: 고등학생 대상인데 중학교 교사)
- L3(이미 구체적인 프롬프트) 비중이 15%로 적고, 수행 단계가 3~4개로 쏠려 있음
- 정량 평가(테스트셋 전체 자동 평가, LLM judge)는 아직 하지 않음

## 라이선스

베이스 모델 [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)는 **Qwen Research License**를 따르며 상업적 사용에 제한이 있습니다. 학습된 어댑터도 이 조건의 영향을 받습니다.
