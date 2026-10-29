# 프롬프트 최적화 SLM (Qwen2.5-3B + LoRA)

사용자 질문을 적절한 역할과 목적으로 생성하고, 사전에 만들어진 카탈로그로부터 적절한 과제와 형식, 제한사항을 라우팅하여 한 문단으로 된 최적화 프롬프트를 생성하는 모델입니다.

- 모델:[Wasteland0103/qwen2.5-3b-prompt-optimizer-lora](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-lora)

## 폴더 구조
```
train.py                모델 학습
inference.py            모델 로드 + 추론(허깅페이스)
data/dataset.jsonl      학습/검증 데이터 1,320개
DATASETS.md             데이터셋 생성 프롬프트와 라벨 카탈로그
eval/evaluate.py        라벨 정확도 평가
eval/ood.jsonl          OOD 테스트셋 49문항
eval/ood2.jsonl         OOD 테스트셋 40문항
app/backend/server.py   FastAPI 로컬 서버
app/frontend/           웹 화면 (Next.js, React, Tailwind )
  app/page.tsx              메인 화면 (입력, 결과 표시, 복사)
  app/layout.tsx            HTML 뼈대
  app/globals.css           전역 스타일
```

## 설치
```bash
pip install -r requirements.txt   # torch는 CUDA 버전에 맞게 설치
```

## 추론
```bash
python inference.py "갤럭시 S25랑 아이폰 17 중에 뭐가 나아? 결론만"
python inference.py --file 요청과본문.txt     # 요청 + 붙여넣은 본문 (자동 분리)
python inference.py                          # 대화형
```
```python
from inference import PromptOptimizer

opt = PromptOptimizer()
r = opt("자취생 첫 전자레인지 추천해줘 20만원 이하로")
print(r["prompt"])
```
출력 예:
```
당신은 생활용품 큐레이터입니다. 전자레인지 추천이 목적이니, 사용자의 요청에 맞는 항목을 추천하세요. 가장 추천하는 항목 N개를 선정하고, 각 항목별 핵심 추천 이유와 특징을 순위/목록 형태로 정리하세요. 성능, 품질, 기능 대비 가격의 합리성(가성비)을 최우선 평가 기준으로 설정하여 추천하세요. 위 지침은 기본값이며, 아래 요청에 숫자(분량, 개수, 예산 등), 대상, 평가 기준, 추가 조건이 명시되어 있으면 그것을 우선하여 모두 반영하세요.

요청: 자취생 첫 전자레인지 추천해줘 20만원 이하로
```

## 데이터셋 (`data/dataset.jsonl`)
```json
{"query": "...", "role": "...", "purpose": "...", "task": "comparison", "format": "F_COMP_TABLE", "constraint": "C_COMP_VERDICT", "source": "template", "split": "train"}
```
| source | split | 개수 | 설명 |
|---|---|---|---|
| template | train | 900 | 템플릿으로 생성한 질문 |
| template | eval | 100 | 위와 같은 분포, task별 20개 층화 추출 |
| natural | train | 320 | 직접 작성한 자연어 질문 (학습 시 2배 오버샘플링) |

데이터 생성에 사용한 프롬프트와 카탈로그는 [DATASETS.md](./DATASETS.md)를 참고하세요.

## 학습

### 학습 설정
| 항목 | 값 |
| :--- | :--- |
| 베이스 모델 | Qwen2.5-3B-Instruct (bf16) |
| LoRA | r=16, alpha=32, dropout 0.05, q/k/v/o/gate/up/down 전체 (학습 파라미터 약 3천만 개, 0.96%) |
| 배치 | batch 1 × grad_accum 16 = 유효 16 |
| 학습률 | 2e-4, cosine, warmup 8 step |
| Loss | assistant(JSON) 토큰에만 계산 (system·질문 토큰은 -100으로 마스킹) |
| 자연어 데이터 | 학습 시 2배 오버샘플링 (템플릿 900개 대비 비중 확대) |

```bash
python train.py models/lora 2e-4 2    # 출력 폴더, lr, epochs
python inference.py "질문" --adapter models/lora
```
12GB GPU 기준 약 30분. batch를 2 이상으로 올리면 VRAM이 공유 메모리로 넘쳐 step당 15~27초로 느려지므로 batch 1 × accum 16을 사용합니다.

### 학습 과정 (3회)
배포된 모델은 **3차 학습** 결과입니다. 회차마다 이전 결과의 문제를 확인하고 데이터를 고쳐 다시 학습했습니다.

| 회차 | 학습 데이터 | epoch | step | 최종 eval loss | in-dist (100) | OOD (49) | OOD2 (40) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 베이스 (학습 전) | - | - | - | - | 20% | - | - |
| 1차 | 템플릿 900 | 3 | 171 | 0.003 | 98%* | 87.8% | 80.0% |
| 2차 | 템플릿 900 (단서 표현 확장) | 2 | 114 | 0.011 | 92% | 85.7% | 77.5% |
| **3차 (배포)** | 템플릿 900 + 자연어 320×2 | 2 | 194 | 0.006 | **97%** | **91.8%** | **90.0%** |

> 정확도 = task, format, constraint 세 항목을 모두 맞힌 비율. 세 회차 모두 JSON 파싱 성공률 100%, 카탈로그 위반 0%.
> \* 2차부터 템플릿 생성기를 바꾸면서 in-dist 평가셋도 새로 뽑았습니다. 1차의 98%는 1차 학습 데이터와 같은 템플릿으로 만든 평가셋에서 나온 점수라 2·3차 점수와 직접 비교할 수 없습니다. 세 회차를 같은 문항으로 비교한 것은 OOD와 OOD2입니다.

**베이스 모델 (학습 전)**
- 파인튜닝 없이 시스템 프롬프트만 주면 세 항목 모두 맞힌 비율이 20%였습니다.
- task는 87% 맞혔지만 format 53%, constraint 30%에 그쳤고, 27%는 해당 task의 카탈로그에 없는 ID를 골랐습니다.

**1차 학습: 템플릿 암기 문제**
- 템플릿 데이터 900개로 3 epoch 학습했습니다. eval loss가 0.003까지 내려가고 in-dist 정확도는 98%였습니다.
- 하지만 처음 보는 표현의 질문(OOD)에서는 80~88%로 떨어졌습니다.
- 원인: 라벨마다 단서 표현이 3~4개뿐이고 문장 골격이 하나로 고정되어 있어서, 모델이 의미 대신 문구 자체를 외웠습니다. 3 epoch도 암기를 부추겼습니다.

**2차 학습: 템플릿 다양화**
- 라벨별 단서 표현을 2배 이상 늘렸습니다 (동의어, 구어체).
- 과제 동사를 확장했습니다 (정리해줘, 골라줘, 짜줘, 뭐가 달라? 등).
- 단서 순서를 섞고, 단서를 주제 앞에 두거나 라벨과 무관한 맥락 문장을 넣는 식으로 문장 골격을 다양화했습니다.
- epoch를 3에서 2로 줄였습니다.
- 결과: OOD의 task 분류는 100%가 됐지만, format과 constraint는 1차와 비슷했습니다 (40문항 중 1문항 차이로, 오차 범위 안).
- 템플릿을 늘리는 것만으로는 한계가 있다고 판단했습니다.

**3차 학습: 자연어 데이터 추가 (최종)**
- 템플릿이 아닌 자연스러운 질문 320개(task별 64개)를 직접 작성했습니다.
- 구어체, 축약형, 단서가 암시만 된 질문, 모호한 질문, 조건이 두 개인 질문을 섞었고, OOD 테스트 문장과는 겹치지 않게 했습니다.
- 템플릿 900개보다 수가 적어서 2배 오버샘플링했습니다.
- 결과: OOD2가 77.5%에서 90.0%로 가장 크게 올랐고, OOD 전체(89문항) 오류는 14개에서 8개로 줄었습니다.

## 평가
| 평가셋 | 문항 | 설명 |
| :--- | :---: | :--- |
| in-dist | 100 | `data/dataset.jsonl`의 `split=eval`. 학습 데이터와 같은 템플릿에서 task별 20개씩 층화 추출 |
| OOD | 49 | `eval/ood.jsonl`. 템플릿 밖의 자연스러운 질문 |
| OOD2 | 40 | `eval/ood2.jsonl`. 2차 학습 이후 새로 만든 공정 테스트셋 (확장한 단서 표현과 겹치지 않게 작성) |

```bash
python eval/evaluate.py                         # 배포 어댑터로 세 평가셋 모두
python eval/evaluate.py --sets ood ood2         # 일부 평가셋만
python eval/evaluate.py --adapter models/lora   # 직접 학습한 어댑터
python eval/evaluate.py --adapter base          # 파인튜닝 전 베이스 모델
```
JSON 파싱률, task/format/constraint 정확도, 세 항목 모두 정답 비율, 카탈로그 위반률과 자주 틀린 라벨을 출력하고, 예측 결과는 `eval/results/`에 저장합니다.

## 웹에서 써 보기 (localhost)
FastAPI 서버 하나가 모델 API와 웹 화면을 함께 띄웁니다.
```bash
cd app/frontend && npm install && npm run build && cd ../..   # 웹 화면 빌드 (처음 한 번)
python app/backend/server.py                                  # http://localhost:8000
```
API만 쓰려면 웹 화면을 빌드하지 않아도 됩니다:
```bash
curl -X POST http://localhost:8000/optimize -H "Content-Type: application/json" -d '{"text": "양자컴퓨터 설명해줘"}'
```
