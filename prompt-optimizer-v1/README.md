# 프롬프트 최적화 SLM (Qwen2.5-3B + LoRA)

사용자 질문을 적절한 역할과 목적으로 생성하고, 사전에 만들어진 카탈로그로부터 적절한 과제와 형식, 제한사항을 라우팅하여 한 문단으로 된 최적화 프롬프트를 생성하는 모델입니다.

- 모델:[Wasteland0103/qwen2.5-3b-prompt-optimizer-lora](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-lora)

## 폴더 구조
```
train.py                모델 학습
inference.py            모델 로드 + 추론(허깅페이스)
data/dataset.jsonl      학습/검증 데이터 1,320개
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

## 학습
```bash
python train.py models/lora 2e-4 2    # 출력 폴더, lr, epochs
python inference.py "질문" --adapter models/lora
```
LoRA r=16, alpha=32, 7개 projection 전체, assistant(JSON) 토큰에만 loss. 12GB GPU 기준 약 30분.

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

## 결과 (task, format, constraint 세 항목 모두 정답 비율)
| 평가셋 | 정확도 |
|---|---|
| in-dist 검증셋 (100문항) | 97.0% |
| OOD 테스트 1 (49문항) | 91.8% |
| OOD 테스트 2 (40문항) | 90.0% |
