# apples-project-prompt-optmization
한국외국어대학교 응용언어학회 APPLES 26-2학기 프롬프트 최적화 모델링 프로젝트 공간입니다.
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)

## 커밋 & 브랜치 규칙(Git Convention)
### 1. 커밋 메시지 컨벤션
`[Type] 한 줄 설명 (#이슈번호)` 형식으로 작성해주세요
| Type | 설명 | 작성 예시 |
| :--- | :--- | :--- |
| **`feat`** | 새로운 기능 / 분석 로직 추가 | `feat: EDA 결측치 처리 함수 구현` |
| **`fix`** | 버그 또는 오류 수정 | `fix: 전처리 스크립트 인덱스 오류 수정` |
| **`docs`** | 문서 수정 (README, 주석 등) | `docs: README에 커밋 컨벤션 추가` |
| **`style`** | 코드 포맷팅, 주석 정리 (로직 변경 없음) | `style: PEP8 코드 스타일 적용` |
| **`refactor`** | 코드 구조 개선 (기능 동작 변경 없음) | `refactor: XGBoost 모델 학습 코드 모듈화` |
| **`chore`** | 설정 파일, 패키지 설치 (`requirements.txt`) | `chore: seaborn 패키지 추가` |

### 2. 브랜치 전략 (Branch Strategy)
* **`main`**: 최종 완성본 및 제출용 (직접 Push 금지 🚫)
* **`dev`**: 개발 통합 브랜치
* **`feature/작업명`**: 기능/분석 단위별 개인 작업 브랜치 (예: `feature/eda`, `feature/modeling`)

### 3. 작업 흐름 (Workflow)
```bash
# 1. dev 최신 코드 동기화
git checkout dev
git pull origin dev

# 2. 개인 작업 브랜치 생성 및 이동
git checkout -b feature/분석내용

# 3. 작업 완료 후 커밋 및 원격 저장소 Push
git add .
git commit -m "feat: 분석 로직 추가"
git push origin feature/분석내용

# 4. GitHub에서 dev 브랜치로 PR(Pull Request) 생성 -> 팀원 확인 후 Merge
```


## 프로젝트 소개(Overview)
- **프로젝트명**:
- **개발 기간**: 26.10.10 ~
- **한 줄 설명**:

## 팀원(Team Members)
| 이름 | 역할 | 전공 | GitHub / Contact |
| :---: | :---: | :--- | :---: |
| **김재우** | 팀장 | ELLT학과 | [@github_id] |
| **박령경** | 팀원 | ELLT학과 | [@github_id] |
| **하은호** | 팀원 | ELLT학과 | [@github_id] |


## 기술 스택(Tech Stack)
| 구분 | 기술 스택 |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Environment** | |
| **Collaboration** | |

## 데이터 세트 소개(Dataset)
- **데이터 출처**: 
- **데이터 크기**: 
- **주요 타겟 변수(Target)**: 
- **주요 피처(Features)**:
  -
  -
  -

## 프로젝트 수행 절차(Workflow)
