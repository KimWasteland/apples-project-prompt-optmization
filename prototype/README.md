# 프롬프트 최적화 SLM (Qwen2.5-3B + LoRA)

사용자 질문 → 라벨 JSON `{role, purpose, task, format, constraint}` → 최적화된 프롬프트.
명세는 `docs/데이터셋 포맷.md`.

## 폴더 구조
```
gen_dataset.py     템플릿 데이터 생성        → data/raw/dataset_v*.jsonl
prep.py            chat 형식 변환 + 90/10 분리 → data/train/{train,eval}_v*.jsonl
prep_natural.py    자연어 데이터 추가(v3)    → data/raw/natural.jsonl, data/train/train_v3.jsonl
train.py           LoRA 학습                 → models/lora_*
evaluate.py        정확도 평가               → results/preds/*_preds_*.jsonl
build_prompt.py    라벨 JSON → 최종 프롬프트 조립 (기본 어댑터 models/lora_v3)
naturalize.py      태그형 지시 → 자연어 한 문단 (규칙 연결 / LM 재작성 + 검증)
server.py          FastAPI 모델 서버 (POST /optimize, GET /health)
web/               Next.js + React + Tailwind 프론트엔드 (Vercel 배포)
scripts/run_train.sh  학습 + in-dist/ood/ood2 평가 일괄 실행
scripts/run_server.sh, run_tunnel.sh  모델 서버 / ngrok 터널 실행

data/raw/     원천 데이터 (템플릿 v1·v2, 직접 작성한 자연어 natural_raw.txt)
data/train/   학습/검증 세트 (v1, v2, v3 — eval_v3는 eval_v2와 동일)
data/test/    OOD 테스트 (ood 49문항, ood2 40문항 공정 테스트)
models/       LoRA 어댑터 v1, v2, v3(최종)
results/      평가 예측, 프롬프트 출력 예시
logs/         학습·평가 로그
```

모든 명령은 프로젝트 루트에서 실행합니다. 환경: `~/anaconda3/envs/test` (Python 3.11, torch cu128).

## 사용 예
```bash
python build_prompt.py "갤럭시 S25랑 아이폰 17 중에 뭐가 나아? 결론만"
python build_prompt.py --file 요청과본문.txt --out result.jsonl
scripts/run_train.sh v4 2e-4 2 _v3
```

## 웹 서비스
```
브라우저 → Vercel (web/, Next.js) → /api/optimize → ngrok → 내 PC:8000 (server.py, FastAPI + 3B 모델)
```
- 서비스 주소: https://prompt-optimizer-one-umber.vercel.app
- 켜기/끄기: `start.bat` / `stop.bat` 더블클릭 (서버 + ngrok, 끄면 GPU 메모리 반환)
- 수동 실행 (터미널 2개): `scripts/run_server.sh` → `scripts/run_tunnel.sh shore-salt-antelope.ngrok-free.dev`
- 비밀키: `.env.server`(백엔드)와 Vercel 환경변수 `API_KEY`가 같아야 함. 백엔드 주소는 Vercel 환경변수 `BACKEND_URL`
- 프론트 수정 후 재배포: `cd web && vercel deploy --prod`
- 출력 모드: natural(규칙 기반 연결, 기본) / tagged(섹션형) / natural_llm(LM 다듬기, 검증 실패 시 natural로 대체)

## 결과 (세 항목 모두 정답 비율)
| | v1 | v2 | v3 |
|---|---|---|---|
| in-dist | 98% | 92% | 97% |
| ood | 87.8% | 85.7% | 91.8% |
| ood2 | 80.0% | 77.5% | 90.0% |
