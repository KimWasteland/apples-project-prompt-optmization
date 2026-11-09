# 프롬프트 최적화 SLM v3 (Qwen2.5-3B + QLoRA)
v2의 섹션 구조에서 성능을 떨어뜨린 부분(역할, {{빈칸}}, 근거 없는 분량 제약)을 빼고, 원래 요청을 프롬프트 끝에 그대로 붙이는 "근거 있는 구조화" 프롬프트를 생성합니다.

- 모델: [Wasteland0103/qwen2.5-3b-prompt-optimizer-v3-lora](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-v3-lora)
- 설계 근거: v2 섹션 제거 실험, 구조화 실험 ([연구일지](../docs/연구일지.md) 2장)

## 폴더 구조
```
utils.py                 이 폴더의 스크립트가 같이 쓰는 코드 (경로, 파일, 통계, GPT/Ollama/보상 모델 호출)
optimizer.py             v3 모델 추론
train.py                 학습
lora_adapter/            직접 학습하면 생기는 어댑터 (git에는 올리지 않음)
test_samples.txt         학습 직후 test 3문항 생성 결과

data/                    학습 데이터
  build_data.py          학습 데이터 만들기 (질문 생성, 정답 작성, 보상 모델로 거르기)
  dataset_v3.jsonl       최종 학습 데이터 865개 (train 757 / val 54 / test 54)
  build/                 데이터를 만드는 중간 결과
    new_inputs.jsonl       새로 만든 질문 (넘기는 예시용, 이미 자세한 질문)
    outputs.jsonl          gpt-4o-mini가 쓴 정답
    dataset_raw.jsonl      거르기 전 데이터
    filter_answers.jsonl   거르기에 쓴 7B 답변
    filter_scores.jsonl    그 답변의 보상 모델 점수

eval/                    평가
  experiments.py         설계 실험 (v3 형식을 정한 섹션 제거 실험, 구조화 실험)
  evaluate.py            v3 평가 (WildChat 평가 문항 선정 포함)
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
python prompt-optimizer-v3/optimizer.py "참고문헌 관리 프로그램 비교해주세요"
```
```python
# prompt-optimizer-v3 폴더 안에서
from optimizer import PromptOptimizer

optimize = PromptOptimizer()
print(optimize("참고문헌 관리 프로그램 비교해주세요"))
```
출력 예:
```
[과제 유형] 비교

[과제] 참고문헌 관리 프로그램을 비교해 주세요.

[맥락 및 가정] 참고문헌 관리 프로그램은 학술 연구와 논문 작성에서 필수적인 도구로 사용됩니다. 일반적으로 사용되는 프로그램에는 Zotero, Mendeley, EndNote 등이 있으며, 각 프로그램의 기능, 사용자 친화성, 무료/유료 여부 등을 포함하여 비교합니다.

[수행 단계]
1. 각 프로그램의 기본 정보 수집 (회사, 출시 연도, 무료/유료 여부)
2. 주요 기능 분석 (원래 참고문헌 추가, 참고문헌 정렬, 인용 형식 변환 등)
3. 사용자 친화성 평가 (UI, 사용 편의성, 교육 자료 제공 여부)
4. 결론 도출 및 추천 프로그램 선정


[원래 요청]
참고문헌 관리 프로그램 비교해주세요
```
(입력: "참고문헌 관리 프로그램 비교해주세요")

## 데이터셋 (`data/dataset_v3.jsonl`)
```json
{"id": "cmp_0002", "task": "비교", "level": "L1", "domain": "교육·학습", "persona": "학생", "tone": "구어체", "input": "토익 토스 뭐가 나음?", "output": "[과제 유형] 비교\n\n[과제] ...\n\n[원래 요청]\n토익 토스 뭐가 나음?", "split": "train", "source": "v2", "filter": {"orig": 6.844, "v3": 5.656, "v3nf": 7.094, "chosen": "v3nf"}}
```
| 필드 | 값 |
|---|---|
| task, level, domain, persona, tone | v2와 같음 (새로 만든 질문은 domain, persona, tone 없음) |
| source | v2(v2 질문), pass(새로 만든 넘기는 예시), l3(새로 만든 이미 자세한 질문), v2+check_fail(GPT가 규칙을 3번 어겨 넘기는 예시로 바꾼 v2 질문) |
| filter | 보상 모델 점수. orig(원문), v3(그대로), v3nf([출력 형식] 뺀 것), chosen(고른 후보) |
| split | train 757 / val 54 / test 54 (val, test는 v2와 같은 문항) |

출력 템플릿:
```
[과제 유형] {task}
[과제]          핵심 지시 1~2문장 (필수)
[맥락 및 가정]  원래 요청의 정보를 빠짐없이 + 부족한 정보는 "가정: ..."
[수행 단계]     번호 목록 2~4단계
[출력 형식]     답의 구조만 (표, 목록, 소제목 등). 도움이 될 때만
[원래 요청]     원문 그대로
```
- [과제] 말고는 실제로 도움이 될 때만 씁니다.
- 역할, {{빈칸}}, 원문에 없는 분량과 톤은 쓰지 않습니다.
- 수학, 코딩, 추출, 추론처럼 구조화가 도움이 안 되는 요청은 `[과제 유형] 기타` + `[원래 요청]` 원문만 냅니다 (넘기는 예시).

### 만드는 과정
```bash
python prompt-optimizer-v3/data/build_data.py inputs      # 넘기는 예시용 질문, 이미 자세한 질문 생성 (gpt-4o-mini)
python prompt-optimizer-v3/data/build_data.py outputs     # 정답 작성 (gpt-4o-mini, 규칙 위반 시 재시도)
python prompt-optimizer-v3/data/build_data.py assemble    # [과제 유형], [원래 요청]을 코드로 붙임
python prompt-optimizer-v3/data/build_data.py answer      # 원문, 후보 2개로 Qwen2.5-7B 답변 (Ollama)
python prompt-optimizer-v3/data/build_data.py score       # 보상 모델 채점 (GPU)
python prompt-optimizer-v3/data/build_data.py filter      # -> data/dataset_v3.jsonl
```
- 후보는 정답 그대로(v3)와 [출력 형식]을 뺀 것(v3nf) 2개입니다.
- 원문 답보다 점수가 높은 후보 중 가장 좋은 것만 남기고, 둘 다 못 이기면 뺍니다.

### 분포 (train 757)
| 정답 형태 | 개수 |
| :--- | :---: |
| 섹션으로 다시 씀, [출력 형식] 없음 | 354 |
| 섹션으로 다시 씀, [출력 형식] 있음 | 300 |
| 넘기는 예시 | 103 |

| 추천 | 설명 | 기획 | 비교 | 기타 | 요약 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 143 | 137 | 127 | 119 | 117 | 114 |

| L1 | L2 | L3 | 질문 길이 (중앙값) | 정답 길이 (중앙값) |
| :---: | :---: | :---: | :---: | :---: |
| 319 | 282 | 156 | 44자 | 378자 |

## 학습

### 학습 설정
| 항목 | 값 |
| :--- | :--- |
| 베이스 모델 | Qwen2.5-3B-Instruct, 4bit 양자화 (Unsloth) |
| LoRA | r=16, alpha=32, dropout 0, q/k/v/o/gate/up/down 전체 |
| 배치 | batch 2 × grad_accum 8 = 유효 16 |
| 학습률 | 2e-4, cosine, warmup 3% |
| epoch | 3 |
| 기타 | max_seq_length 2048, adamw_8bit, bf16, assistant 응답 토큰에만 loss 계산, seed 42 |
| 시스템 프롬프트 | v2와 같음 |

```bash
pip install unsloth trl datasets
python prompt-optimizer-v3/train.py     # -> prompt-optimizer-v3/lora_adapter/, test_samples.txt
```

## 평가 결과 (원문 대비)
답변 모델 Qwen2.5 3B, 7B, 14B (Ollama). 채점은 모두 원래 질문 기준입니다.

| 평가 | 3B | 7B | 14B |
| :--- | :--- | :--- | :--- |
| WildChat 300 GPT 판정 승률 | 48.3% [43.7%, 53.2%], p 0.688 | **55.0%** [50.4%, 59.3%], p 0.044 | **59.2%** [54.9%, 63.3%], p 0.000 |
| WildChat 300 보상 모델 Δ | -1.04 [-1.52, -0.56], p 0.000 | -0.32 [-0.70, +0.07], p 0.040 | +0.11 [-0.15, +0.38], p 0.454 |
| test 54 보상 모델 Δ | +0.51 [-0.63, +1.70], p 0.633 | +0.21 [-0.75, +1.26], p 0.816 | +0.08 [-0.67, +0.92], p 0.567 |
| KoMT 1턴 gpt-4o 점수 Δ | -0.28 [-0.87, +0.30], p 0.308 | - | - |
| IFEval-Ko 문항 strict 정답률 | -6.7%p, p 0.019 | -6.7%p, p 0.008 | -8.8%p, p 0.000 |

- 대괄호는 95% 부트스트랩 신뢰구간입니다. p값은 점수 차이는 윌콕슨, GPT 판정은 부호 검정, IFEval은 McNemar로 구했습니다.
- 굵은 칸은 원문보다 유의하게 나은 경우입니다.
- 결과 파일은 `results/v3/`, 결과표는 `python prompt-optimizer-v3/eval/evaluate.py report`로 다시 만들 수 있습니다.

## 알려진 한계
- 답변 모델이 3B면 효과가 없고, 보상 모델로는 오히려 손해입니다.
- 지시가 붙은 요청(IFEval-Ko)을 구조화하면 원래 조건이 흐려져 세 모델 모두 정답률이 유의하게 떨어집니다.
- 보상 모델로 후보를 고르는 거르기 방식은 잡음을 고르는 경우가 많았습니다 (승자의 저주).
- 바꾼 프롬프트의 답이 원문 답보다 길어서, 7B와 14B의 GPT 판정 승리가 답의 질 때문인지 길이 때문인지는 가를 수 없습니다.
