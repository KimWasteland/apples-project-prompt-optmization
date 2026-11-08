# IFEval-Ko: 한국어 LLM Instruction-Following 벤치마크

> 이 데이터셋은 구글 리서치팀의 [IFEval](https://huggingface.co/datasets/google/IFEval/) 데이터셋을 기반으로 제작되었습니다.

[영어 원문 README](https://huggingface.co/datasets/allganize/IFEval-Ko/)

`IFEval-Ko`는 Google의 오픈소스 **IFEval** 벤치마크를 한국어로 변형하여 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 프레임워크를 통해 
대규모 언어 모델(LLM)의 한국어 Instruction-Following 능력을 평가할 수 있도록 제작된 벤치마크입니다.

## 데이터셋 상세 정보

- **원본 출처**: [google/IFEval](https://huggingface.co/datasets/google/IFEval/)
- **변형 제작자**: [Allganize Inc. LLM TEAM](https://www.allganize.ai/) | Keonmo Lee
- **저장소**: [allganize/IFEval-Ko](https://huggingface.co/datasets/allganize/IFEval-Ko)
- **언어**: 한국어
- **번역 도구**: GPT-4o
- **라이선스**: 원본 [google/IFEval](https://huggingface.co/datasets/google/IFEval/) 라이선스 준수
- **벤치마크 도구**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

## 벤치마크 점수

![plot](https://huggingface.co/datasets/allganize/IFEval-Ko/resolve/main/benchmark-score-dot-plot.png)

## 사용 방법

`lm-evaluation-harness` 저장소를 클론한 후, `lm_eval/tasks` 폴더 안에 `ifeval_ko` 폴더를 생성하세요.

```bash
# lm-evaluation-harness 및 필요 라이브러리 설치
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
pip install langdetect immutabledict

# Hugging Face에서 task 파일 다운로드
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='allganize/IFEval-Ko',
    repo_type='dataset',
    local_dir='lm_eval/tasks/',
    allow_patterns='ifeval_ko/*',
    local_dir_use_symlinks=False
) "
```

***사용 전 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 공식 문서를 참고하세요.***

### Hugging Face Transformers로 평가

```bash
lm_eval --model hf \
    --model_args pretrained={HF_MODEL_REPO} \
    --tasks ifeval_ko \
    --device cuda:0 \
    --batch_size 8
```
예시) {HF_MODEL_REPO} = google/gemma-3-4b-it

### vLLM으로 평가

vLLM 호환 백엔드 설치:

```bash
pip install lm-eval[vllm]
```

그 후 평가 실행:

```bash
lm_eval --model vllm \
    --model_args pretrained={HF_MODEL_REPO},trust_remote_code=True \
    --tasks ifeval_ko
```

---

## 원본 IFEval 대비 수정 사항

### 데이터 변형

- **번역**: 원문 구조를 유지하는 맞춤형 프롬프트를 활용하여 **gpt-4o** 모델로 번역
- **제거된 항목**:  
  - 대소문자 구분(`change_case`) 관련 84개 문항  
  - 알파벳 기반(`letter_frequency`) 관련 28개 문항  
  - 번역 오류 또는 문화적으로 부적합한 프롬프트, 실행시 문제가 발생하는 문항
- **단위 변환**:  
  - 갤런 → 리터  
  - 피트/인치 → 미터/센티미터  
  - 달러 → 원 (USD:KRW ≈ 1:1500)
- **표준화 작업**:  
  - 헤더  \<\<Title\>\>, \<\<title\>\> 를 모두 \<\<제목\>\> 으로 통일
  - 답변 문체를 통일감 있게 정리

### 코드 변경

- 번역된 instruction 옵션 적용:
  - `instruction._CONSTRAINED_RESPONSE_OPTIONS`
  - `instruction._ENDING_OPTIONS`
- 스코어링 클래스 수정:
  - `KeywordChecker`, `KeywordFrequencyChecker`, `ParagraphFirstWordCheck`, `KeySentenceChecker`, `ForbiddenWords`, `RepeatPromptThenAnswer`, `EndChecker`
  - `unicodedata.normalize('NFC', ...)`를 적용하여 정규화
  - 필드 누락 시 예외 발생하도록 키워드 생성 로직 수정 (fallback 제거)
- 문장 수 세기(`count_sentences()`) 로직을 수정하여 `nltk` 의존성 제거

---

## 평가 지표

자세한 내용은 [IFEval 논문](https://arxiv.org/pdf/2311.07911) 참고:

### Strict vs. Loose Accuracy

- **Strict**: 모델이 지시사항을 변형 없이 정확히 따랐는지 평가
- **Loose**: 응답에 3가지 변환을 적용한 후 평가
  1. 마크다운 기호 (`*`, `**`) 제거
  2. 첫 번째 문장 (예: "Here is your response:") 제거
  3. 마지막 문장 (예: "Did that help?") 제거

  변환 조합 8개 중 하나라도 일치하면 정답 처리

### Prompt-level vs. Instruction-level

- **Prompt-level**: 하나의 프롬프트에 있는 모든 instruction을 정확히 수행해야 True
- **Instruction-level**: instruction 별로 독립적으로 평가하여 세분화된 지표 제공

제작:  
Allganize LLM TEAM  
[**Keonmo Lee (\uC774\uAC74\uBAA8)**](https://huggingface.co/whatisthis8047)


### Citation Information
```
@misc{zhou2023instructionfollowingevaluationlargelanguage,
      title={Instruction-Following Evaluation for Large Language Models}, 
      author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
      year={2023},
      eprint={2311.07911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07911}, 
}
```