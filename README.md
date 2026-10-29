# apples-project-prompt-optmization
한국외국어대학교 응용언어학회 APPLES 26-2학기 프롬프트 최적화 모델링 프로젝트 공간입니다.

## 프로젝트 소개(Overview)
- **프로젝트명**: Learning to Ask Better: SLM-Based Prompt Optimization for Improving Answer Quality
- **개발 기간**: 26.08.23 ~
- **한 줄 설명**: 프롬프트 최적화를 SLMs(Small Language Models) 단위에서 수행하기

## 팀원(Team Members)
| 이름 | 역할 | 전공 | GitHub / Contact |
| :---: | :---: | :--- | :---: |
| **김재우** | 팀장 | ELLT학과 | [@KimWasteland](https://github.com/KimWasteland)|
| **박령경** | 팀원 | ELLT학과 | [@RyeongKyeong] |
| **하은호** | 팀원 | ELLT학과 | [@hahoyu07-cmd] |

## 기술 스택(Tech Stack)
| 구분 | 기술 스택 |
| :--- | :--- |
| **Language**      | Python 3.10+, TypeScript |
| **Base Model**    | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
| **Fine-tuning**   | **v1:** LoRA (bf16) — PyTorch, Transformers, PEFT <br> **v2:** QLoRA (4bit) — Unsloth, TRL, PEFT |
| **Serving / Demo**| **Backend:** FastAPI, Uvicorn <br> **Frontend:** Next.js 16, React 19, Tailwind CSS 4 (v1) / HTML (v2) |
| **Model Hub**     | Hugging Face ([v1](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-lora), [v2](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora)) |
| **Environment**   | **OS:** Windows 10 / 11 <br> **Hardware:** NVIDIA GPU RTX5070 <br> **Environment Tool:** Conda |
| **Collaboration** | **Version Control:** Git, GitHub|

## 모델 버전(Versions)
| 구분 | v1 | v2 |
| :--- | :--- | :--- |
| **접근 방식** | 질문 > 역할·목적 생성 + 사전 카탈로그에서 과제/형식/제한사항 라우팅 | 원본 프롬프트에서 개선 프롬프트를 직접 생성 |
| **출력 형태** | 한 문단 프롬프트 | 섹션 구조 프롬프트 ([역할] / [과제] / [맥락 및 가정] / [수행 단계] / [출력 형식]) |
| **학습 방식** | LoRA (bf16), r=16, alpha=32 | 4bit QLoRA (Unsloth), r=16, alpha=32 |
| **데이터셋** | 1,320개 (템플릿 1,000 + 직접 작성 320) | 원본 개선 쌍 1,031개 (train 923 / val 54 / test 54) |
| **학습 시간** | 약 30분 (12GB GPU) | 약 13분 (12GB GPU) |
| **데모** | FastAPI + Next.js | FastAPI + HTML |
| **모델** | [HF 링크](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-lora) | [HF 링크](https://huggingface.co/Wasteland0103/qwen2.5-3b-prompt-optimizer-v2-lora) |
| **상세 문서** | [prompt-optimizer-v1](./prompt-optimizer-v1/README.md) | [prompt-optimizer-v2](./prompt-optimizer-v2/README.md) |

## 예시(v1, Before / After)
**Before**
```
자취생 첫 전자레인지 추천해줘 20만원 이하로
```
**After**
```
당신은 생활용품 큐레이터입니다. 전자레인지 추천이 목적이니, 사용자의 요청에 맞는 항목을 추천하세요. 가장 추천하는 항목 N개를 선정하고, 각 항목별 핵심 추천 이유와 특징을 순위/목록 형태로 정리하세요. 성능, 품질, 기능 대비 가격의 합리성(가성비)을 최우선 평가 기준으로 설정하여 추천하세요. 위 지침은 기본값이며, 아래 요청에 숫자(분량, 개수, 예산 등), 대상, 평가 기준, 추가 조건이 명시되어 있으면 그것을 우선하여 모두 반영하세요.

요청: 자취생 첫 전자레인지 추천해줘 20만원 이하로
```
## 예시(v2, Before / After)
**Before**
```
참고문헌 관리 프로그램 비교해주세요
```
**After**
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

## 프로젝트 수행 절차(Workflow)
* **8월: 선형대수학 및 머신러닝 기초**
  * 벡터의 정의, 행렬의 곱셈, 선형가중결합 및 선형변환
  * 선형회귀 및 로지스틱 회귀 학습
* **9월: 딥러닝 및 트랜스포머**
  * **딥러닝:** 학습 최적화 전략 학습 및 실습 (정규화, 소프트맥스 회귀)
  * **트랜스포머:** 학습 및 실습 (Seq2Seq 모델, 어텐션 메커니즘)
* **10월: 개발 기간**
  * 가설 및 파이프라인 구성, 데이터 세트 생성
  * 프로토타입 완성
