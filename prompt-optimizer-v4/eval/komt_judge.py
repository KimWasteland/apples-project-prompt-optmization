"""KoMT-Bench 공식 채점 코드를 실행한다. evaluate.py의 komt-judge가 부르는 파일이라 직접 실행할 일은 없다.

  python prompt-optimizer-v4/eval/komt_judge.py judge qwen2.5-3b__v4     # gpt-4o로 채점 (공식 gen_judgment.py)
  python prompt-optimizer-v4/eval/komt_judge.py penalty qwen2.5-3b__v4   # 한국어가 아닌 답은 0점 처리 (공식 detector.py)

공식 코드는 고치지 않고, 실행할 때 아래 네 가지만 바꾼다.
  1. 심사 모델 허용 목록에 gpt-4o 이름을 넣는다 (공식은 gpt-4-0613).
  2. 실행 중에 나오는 확인 질문(input)에 자동으로 엔터를 친다.
  3. 수학, 추론, 코딩은 심사 모델 이름으로 참고 답안을 찾는다. 공식 GPT-4 참고 답안을 gpt-4o 이름으로도 찾게 한다.
  4. 언어 감지기가 1순위 언어만 돌려주게 한다 (max_results=1).
     요즘 mediapipe는 확률 0인 2순위 언어까지 돌려주는데, 공식 코드는 마지막 결과를 답의 언어로 써서
     한국어 답도 한국어가 아니라고 판정하기 때문이다.
"""
import builtins
import os
import runpy
import sys
from pathlib import Path

FASTCHAT = Path(__file__).resolve().parents[2] / "benchmarks" / "KoMT-Bench" / "FastChat"
KOMT_BENCH = FASTCHAT / "fastchat" / "llm_judge"
JUDGE_MODEL = "gpt-4o-2024-08-06"

sys.path.insert(0, str(FASTCHAT))   # fastchat 패키지를 설치하지 않아도 불러올 수 있게


def judge(model_id):
    import fastchat.llm_judge.common as judge_common
    import fastchat.model.model_adapter as model_adapter

    # 1. gpt-4o를 심사 모델로 허용
    model_adapter.OPENAI_MODEL_LIST = model_adapter.OPENAI_MODEL_LIST + (JUDGE_MODEL,)

    # 2. 확인 질문에 자동으로 엔터
    builtins.input = lambda *args: ""

    # 3. 참고 답안을 gpt-4o 이름으로도 찾게 한다
    original_load = judge_common.load_model_answers

    def load_with_gpt4o_name(directory):
        answers = original_load(directory)
        if "reference_answer" in directory:
            answers[JUDGE_MODEL] = answers["gpt-4"]
        return answers

    judge_common.load_model_answers = load_with_gpt4o_name

    # 공식 스크립트는 자기 폴더에서 실행해야 한다
    os.chdir(KOMT_BENCH)
    sys.path.insert(0, str(KOMT_BENCH))
    sys.argv = ["gen_judgment.py", "--mode", "single", "--judge-model", JUDGE_MODEL,
                "--model-list", model_id, "--parallel", "1"]
    runpy.run_path("gen_judgment.py", run_name="__main__")


def penalty(model_id):
    folder = KOMT_BENCH / "data" / "mt_bench" / "model_judgment"
    os.chdir(folder)
    sys.path.insert(0, str(folder))

    from mediapipe.tasks import python
    from mediapipe.tasks.python import text
    import detector

    # 4. 1순위 언어만 돌려주는 언어 감지기
    options = text.LanguageDetectorOptions(base_options=python.BaseOptions(model_asset_path="detector.tflite"),
                                           max_results=1)
    language_detector = text.LanguageDetector.create_from_options(options)
    detector.penalty_to_en_answer(model_id, ".", language_detector)


if __name__ == "__main__":
    command, model_id = sys.argv[1], sys.argv[2]
    if command == "judge":
        judge(model_id)
    elif command == "penalty":
        penalty(model_id)
