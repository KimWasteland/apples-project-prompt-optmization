# 프롬프트 최적화 SLM v4 (Qwen2.5-3B + QLoRA)
짧고 모호한 요청을 원래 정보와 말투를 그대로 살린 채, 좋은 답에 필요한 내용을 함께 부탁하는 한 문단의 프롬프트로 다시 씁니다. 정답이 하나로 정해진 요청은 원문 그대로 둡니다.

- 모델: [Wasteland0103/qwen2.5-3b-prompt-optimizer-v4-lora](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-v4-lora)
- v3와 달라진 점: 섹션 헤더 대신 사람이 실제로 쓰는 한 문단. v3 데이터에서 이득이 난 60자 이하 요청만 다시 씀.

## 폴더 구조
```
utils.py                 이 폴더의 스크립트가 같이 쓰는 코드 (경로, 파일, 통계, GPT/Ollama/보상 모델 호출)
optimizer.py             v4 모델 추론 + v4 규칙 (gate, faithful)
train.py                 학습
lora_adapter/            직접 학습하면 생기는 어댑터 (git에는 올리지 않음)

data/                    학습 데이터
  build_data.py          학습 데이터 만들기 (질문 생성, 가이드, 다시 쓰기)
  dataset_v4.jsonl       최종 학습 데이터 1,075개 (train 1,014 / val 31 / test 30)
  build/                 데이터를 만드는 중간 결과
    new_inputs.jsonl       새로 만든 짧은 질문
    guides.jsonl           요청마다 gpt-4o-mini가 쓴 가이드 (갈림길, 다룰 내용)
    rewrites.jsonl         가이드를 녹여 다시 쓴 프롬프트

eval/                    평가
  evaluate.py            v4 평가
  komt_judge.py          KoMT-Bench 공식 채점 코드 실행 (evaluate.py가 부른다)
```

## 설치
```bash
pip install torch transformers peft accelerate   # torch는 CUDA 버전에 맞게 https://pytorch.org 에서 먼저 설치
pip install unsloth trl datasets                 # 학습할 때만
```

## 추론
저장소 루트에서 실행합니다. 어댑터는 Hugging Face Hub에서 내려받습니다.
```bash
python prompt-optimizer-v4/optimizer.py "초생달이 뭐야"
```
```python
# prompt-optimizer-v4 폴더 안에서
from optimizer import load_optimizer

optimize = load_optimizer()   # gate()를 통과한 질문만 모델로 다시 쓰고, faithful() 검사에 걸리면 원문을 돌려준다
print(optimize("참고문헌 관리 프로그램 비교해주세요"))
```
출력 예:
```
참고문헌 관리 프로그램 비교해주세요. 사용자 친화성, 기능 다양성, 가격 정책에 대해 각각 어떤 차이가 있는지 알려주시면 좋겠습니다.
```
(입력: "참고문헌 관리 프로그램 비교해주세요")
```
초생달이 뭐야? 형성 과정이나 주기, 그리고 관측 가능성에 대해서도 설명해줘.
```
(입력: "초생달이 뭐야", 반말은 반말로)

### 추론 과정
1. `gate()`: 다시 쓸 대상인지 고릅니다. 60자 이하 한 줄이고, 분량(숫자 + 자, 줄, 개 등), 형식(표로, 영어로, 간단히 등), 수식, 코드가 없는 요청만 모델에 넣습니다. 나머지는 원문 그대로 보냅니다.
2. 모델이 한 문단으로 다시 씁니다 (repetition_penalty 1.0).
3. `faithful()`: 출력을 검사합니다. 아래 중 하나라도 걸리면 원문을 그대로 보냅니다.
   - 줄바꿈, 괄호, 헤더 흔적이 있음
   - 원문에 없던 한자, 가나가 섞임
   - 문장 수가 1~4 밖
   - 말투가 바뀜
   - 원문 글자 보존이 80% 미만이거나 원문의 영어 용어, 숫자가 빠짐
   - 원문보다 170자 넘게 길어짐

## 데이터셋 (`data/dataset_v4.jsonl`)
```json
{"id": "cmp_0001", "task": "비교", "level": "L1", "input": "파이썬 자바 차이", "split": "train", "system": "너는 프롬프트 최적화기다. ...", "output": "파이썬과 자바의 차이에 대해 알고 싶습니다. 문법의 간결성, 타입 시스템, 생태계 및 라이브러리 지원에 대해 각각 어떤 차이가 있는지 설명해 주실 수 있나요?"}
```
| 필드 | 값 |
|---|---|
| task | 설명, 추천, 기획, 비교, 요약, 기타 |
| level | L1, L2, L3 (v2와 같음) |
| system | v4 시스템 프롬프트 (행마다 들어 있음) |
| split | train 1,014 / val 31 / test 30 |

### 만드는 과정
```bash
python prompt-optimizer-v4/data/build_data.py inputs      # 짧은 요청 512개 새로 생성 (gpt-4o). 평가 문항과 비슷한 것은 제외
python prompt-optimizer-v4/data/build_data.py guides      # 요청마다 가이드 작성 (gpt-4o-mini)
python prompt-optimizer-v4/data/build_data.py rewrite     # 가이드를 원문에 녹여 한 문단으로 (gpt-4o-mini)
python prompt-optimizer-v4/data/build_data.py assemble    # -> data/dataset_v4.jsonl
```
- 가이드는 답이 상황에 따라 갈리는 지점(갈림길) 한 줄과, 이 요청에서만 나오는 다룰 내용 2~4개입니다. "장단점", "비용"처럼 어디에나 붙는 말은 금지했습니다.
- 요약할 글이 없는 요약, 정답이 하나인 요청은 원문 그대로 두는 예시로 썼습니다. 코드 작성과 개념 질문은 다시 쓰도록 규칙으로 고정했습니다.
- 비용, 일정, 최신 정보처럼 지어낼 위험이 큰 요청과 역할을 혼동한 출력은 뺐습니다.
- 다시 쓸 때 원문 말투를 알려 주고, `faithful()`에 걸리면 한 번 더 시도하고, 또 걸리면 뺐습니다.

### 분포 (train 1,014)
| 출처 | 개수 |
| :--- | :---: |
| v2 데이터셋 질문 | 471 |
| 새로 만든 짧고 모호한 질문 | 438 |
| v3의 넘기는 예시 질문 | 57 |
| v3의 이미 자세한 질문 | 48 |

| 정답 형태 | 개수 |
| :--- | :---: |
| 한 문단으로 다시 씀 | 902 |
| 원문 그대로 | 112 |

| 설명 | 추천 | 기타 | 기획 | 비교 | 요약 | 그 밖 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 292 | 270 | 136 | 117 | 113 | 63 | 23 |

| L1 | L2 | L3 | 질문 길이 (중앙값) | 정답 길이 (중앙값) |
| :---: | :---: | :---: | :---: | :---: |
| 735 | 231 | 48 | 15.5자 | 86자 |

## 학습

### 학습 설정
v3와 같습니다.

| 항목 | 값 |
| :--- | :--- |
| 베이스 모델 | Qwen2.5-3B-Instruct, 4bit 양자화 (Unsloth) |
| LoRA | r=16, alpha=32, dropout 0, q/k/v/o/gate/up/down 전체 |
| 배치 | batch 2 × grad_accum 8 = 유효 16 |
| 학습률 | 2e-4, cosine, warmup 3% |
| epoch | 3 |
| 기타 | max_seq_length 2048, adamw_8bit, bf16, assistant 응답 토큰에만 loss 계산, seed 42 |

```bash
pip install unsloth trl datasets
python prompt-optimizer-v4/train.py     # -> prompt-optimizer-v4/lora_adapter/
```

### Loss
- train loss는 약 0.37까지 내려갔습니다.
- eval loss는 1.9 epoch 근처에서 0.693으로 가장 낮았고, 끝에서 0.706이었습니다.

## 평가 결과 (원문 대비)
답변 모델 Qwen2.5 3B, 7B, 14B (Ollama). 채점은 모두 원래 질문 기준입니다.

| 평가 | 3B | 7B | 14B |
| :--- | :--- | :--- | :--- |
| WildChat 300 GPT 판정 승률 | 47.7% [44.3%, 51.2%], p 0.191 (n=299) | **56.0%** [52.7%, 59.3%], p 0.001 | **55.9%** [52.4%, 59.4%], p 0.001 (n=270) |
| WildChat 300 보상 모델 Δ | -0.47 [-0.80, -0.15], p 0.019 | +0.07 [-0.21, +0.35], p 0.477 | +0.15 [-0.05, +0.35], p 0.253 |
| test 54 보상 모델 Δ | -0.42 [-1.04, +0.16], p 0.568 | +0.00 [-0.70, +0.72], p 0.595 | +0.03 [-0.40, +0.44], p 0.364 |
| KoMT 1턴 gpt-4o 점수 Δ | +0.12 [-0.06, +0.31], p 0.233 | - | - |

- 대괄호는 95% 부트스트랩 신뢰구간입니다. p값은 점수 차이는 윌콕슨, GPT 판정은 부호 검정으로 구했습니다.
- 굵은 칸은 원문보다 유의하게 나은 경우입니다.
- 14B GPT 판정은 OpenAI 크레딧이 떨어져 300문항 중 270문항까지만 했습니다. 3B는 판정 형식이 깨진 1문항을 빼서 299문항입니다.
- KoMT는 80문항 중 4개만 바뀌어 원문과 사실상 같습니다. IFEval-Ko는 바뀐 문항이 0개라 뺐습니다.
- 결과 파일은 `results/v4/`, 결과표는 `python prompt-optimizer-v4/eval/evaluate.py report`로 다시 만들 수 있습니다.

## 알려진 한계
- 답변 모델이 3B면 효과가 없고, 보상 모델로는 손해입니다.
- 덧붙인 내용이 사용자 의도와 어긋나는 경우가 많습니다 (예: 타격 자세 질문에 "스트레스 관리").
- 원문에 없던 "자세히", "구체적으로"를 붙인 문항은 승률이 더 낮았습니다.
- 짧은 질문만 보고 덧붙이는 내용은 모델의 사전지식일 뿐이라, 답변 모델이 모르는 새 정보가 생기지 않습니다.
- 바꾼 프롬프트의 답이 원문 답보다 길어서, 7B와 14B의 GPT 판정 승리가 답의 질 때문인지 길이 때문인지는 가를 수 없습니다.
