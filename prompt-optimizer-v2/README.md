# 프롬프트 최적화 SLM v2 (Qwen2.5-3B + QLoRA)
짧고 모호한 프롬프트를 역할 명시, 모호성 해소, 구조화, 출력 형식 명시를 적용하여 최적화된 프롬프트를 생성합니다.

- 모델: [Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora)

## 폴더 구조
```
train.py            QLoRA 학습 (Unsloth + TRL, data/dataset.jsonl → outputs/)
inference.py        모델 로드 + 추론 (Hugging Face Hub의 어댑터를 내려받아 사용)
data/dataset.jsonl  원본 → 개선 프롬프트 쌍 1,031개
app/server.py       로컬 데모 서버 (FastAPI, POST /generate 스트리밍 + 웹 화면)
app/index.html      웹 화면 (테스트셋 예시, 파인튜닝 전/후 비교, 섹션 구조 체크)
```

## 설치
```bash
pip install -r requirements.txt   # torch는 CUDA 버전에 맞게 https://pytorch.org 에서 먼저 설치
```

## 추론
```bash
python inference.py "블록체인 설명해줘"
python inference.py --file 원본프롬프트.txt
python inference.py                          # 대화형
```
```python
from inference import PromptOptimizer

opt = PromptOptimizer()
print(opt("크리스마스 프로모션 기획해줘"))
```
출력 예:
```
[과제 유형] 비교

[역할]
당신은 학술지 논문 작성과 연구실 문헌관리를 지도해 온 연구자입니다.

[과제]
논문 작성 중 사용할 수 있는 참고문헌 관리 프로그램을 기능, 가격, 지원 언어, 한국어 텍스트 인식 등으로 비교하고 추천해 주세요.

[맥락 및 가정]
- 비교 대상: {{비교할 프로그램}} (없으면 일반적인 3개를 가정)
- 작업 환경: {{Windows/Mac/Linux}}
- 필요 기능: {{자동 인덱싱, 한국어 텍스트 인식, 출처 확인, PDF 저장, 표 분석}}

[수행 단계]
1. 비교 항목별로 프로그램마다 점수를 매깁니다.
2. 한국어 사용자의 불편함(텍스트 인식 정확도)을 따로 강조합니다.
3. 최종 추천 순서를 제시합니다.

[출력 형식]
- 형식: 비교 표 + 추천 목록(순위 / 추천 이유 / 적합한 상황)
- 분량: 표 4열 + 추천 3줄 이내
- 톤: 존댓말, 객관적인 검토체
- 반드시 포함: 무료 버전의 한계, 최신 버전 정보는 공식 사이트에서 확인하라는 문구
```
(입력: "참고문헌 관리 프로그램 비교해주세요")

## 웹에서 써 보기 (localhost)
```bash
python app/server.py     # http://localhost:7860
```
- 테스트셋 예시를 불러오거나 직접 입력해 최적화
- "파인튜닝 전과 비교": 같은 입력을 어댑터를 끈 베이스 모델로도 생성해 나란히 비교
- 출력이 정해진 섹션 구조를 지켰는지 자동 체크

## 데이터셋 (`data/dataset.jsonl`)
```json
{"id": "cmp_0001", "task": "비교", "level": "L1", "domain": "IT·개발", "persona": "개발자", "tone": "반말", "input": "파이썬 자바 차이", "output": "[과제 유형] 비교\n\n[역할]\n...", "split": "train"}
```
| 필드 | 값 |
|---|---|
| task | 설명, 요약, 기획, 비교, 추천, 기타 |
| level | L1(키워드 수준), L2(일부 정보 있음), L3(이미 구체적 → 최소 수정) |
| domain | IT·개발, 교육·학습, 비즈니스·마케팅, 금융·경제, 건강·의료, 여행, 요리·생활, 학술·연구, 법·행정, 문화·엔터, 커리어, 육아·관계 |
| persona | 학생, 직장인, 개발자, 자영업자, 교사, 연구자, 주부, 시니어, 취준생, 프리랜서 |
| tone | 반말, 존댓말, 구어체, 영어혼용 |
| split | train 923 / val 54 / test 54 (task × level별 층화 분할) |

출력 템플릿:
```
[과제 유형] {task}
[역할]          구체적 직무·경력·관점
[과제]          명확해진 핵심 지시 1~2문장
[맥락 및 가정]  원본에 있던 정보 / 가정: 합리적 기본값 / 사용자만 알 수 있는 정보는 {{빈칸}}
[원문]          입력에 원문이 있을 때만, 글자 그대로
[수행 단계]     번호 목록 2~5단계
[출력 형식]     - 형식: / - 분량: / - 톤: / - 반드시 포함:
```

## 학습

### 학습 설정
| 항목 | 값 |
| :--- | :--- |
| 베이스 모델 | Qwen2.5-3B-Instruct, 4bit 양자화 (Unsloth) |
| LoRA | r=16, alpha=32, dropout 0, q/k/v/o/gate/up/down 전체 |
| 배치 | batch 2 × grad_accum 8 = 유효 16 |
| 학습률 | 2e-4, cosine, warmup 3% |
| epoch | 3 (174 step) |
| 기타 | max_seq_length 2048, adamw_8bit, bf16, assistant 응답 토큰에만 loss 계산 |
| 데이터 분할 | train 923 / val 54 / test 54 (task × level별 층화, seed 42) |

```bash
pip install unsloth trl datasets
python train.py --max_steps 5     # 스모크 테스트
python train.py                   # 본 학습 (3 epoch)
python train.py --resume outputs/qwen2.5-3b-promptopt/checkpoint-58   # 중단된 학습 이어서
python inference.py "질문" --adapter outputs/qwen2.5-3b-promptopt/lora_adapter
```
12GB GPU 기준 약 13분 (745초).

### 학습 과정
- 처음 설정(batch 4 × grad_accum 4)으로 학습하던 중 92 step에서 GPU 메모리 부족(OOM)이 발생했습니다.
- 유효 배치 16은 유지하면서 batch 2 × grad_accum 8로 조정해 학습을 완료했습니다.
- VRAM이 더 부족하면 `--batch_size 1 --grad_accum 16`으로 줄이세요.

### Loss 추이
| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 174 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| eval loss | 1.229 | 1.080 | 1.021 | 0.993 | 0.970 | 0.956 | 0.965 | 0.958 | 0.958 |

- train loss는 2.10(step 5)에서 약 0.76(step 170)으로 내려갔습니다.
- eval loss는 120 step 이후 0.96 근처에서 수렴했고, 다시 오르지 않아 과적합 징후는 없습니다.
- v1(라벨 JSON 약 50토큰)과 달리 v2는 긴 프롬프트 전체를 생성하므로 loss의 절대값이 높습니다. 두 버전의 loss를 직접 비교하면 안 됩니다.

### 알려진 한계
- 입력에 없는 요구 사항을 지어내는 경우가 있습니다.
- 역할이 과제와 어긋나는 경우가 있습니다 (예: 고등학생 대상인데 중학교 교사).
- L3(이미 구체적인 프롬프트) 데이터가 15%로 적고, 수행 단계가 3~4개로 쏠려 있습니다.
