"""프롬프트 최적화 SLM 추론 (Qwen2.5-3B-Instruct + LoRA).

사용자 질문 -> 모델이 라벨 JSON {role, purpose, task, format, constraint} 생성
-> 카탈로그 문장을 규칙 기반으로 한 문단으로 이어 최종 프롬프트를 만든다.

사용:
  python inference.py "갤럭시 S25랑 아이폰 17 중에 뭐가 나아? 결론만"
  python inference.py --file 요청과본문.txt          # 요청 + 붙여넣은 본문 (자동 분리)
  python inference.py                               # 대화형
  python inference.py "질문" --adapter models/lora  # 직접 학습한 로컬 어댑터 사용
"""
import argparse, json, re

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "Wasteland0103/qwen2.5-3b-prompt-optimizer-lora"

# ------------------------------------------------------------------ 카탈로그 (task별, 첫 번째가 디폴트)
# {task:(format, constraint)}
CATALOG = {
    "summarization": (["F_SUM_BULLETS", "F_SUM_PROSE", "F_SUM_KEYWORD", "F_SUM_TIMELINE"],
                      ["C_SUM_LENGTH_SHORT", "C_SUM_LENGTH_LONG", "C_SUM_NO_JARGON", "C_SUM_FACT_ONLY", "C_SUM_ACTION_ITEM"]),
    "comparison": (["F_COMP_TABLE", "F_COMP_PRO_CON", "F_COMP_VENN"],
                   ["C_COMP_BALANCED", "C_COMP_VERDICT", "C_COMP_DIFF_ONLY", "C_COMP_COST_EFFICIENCY", "C_COMP_SPECIFIC_METRIC"]),
    "explanation": (["F_EXP_STEPWISE", "F_EXP_QA", "F_EXP_DEEPDIVE"],
                    ["C_EXP_EASY_ELI5", "C_EXP_TECHNICAL_DEEP", "C_EXP_VISUAL_ASCII", "C_EXP_REAL_WORLD_CASE", "C_EXP_MISCONCEPTION"]),
    "recommendation": (["F_REC_TOPN", "F_REC_SITUATIONAL", "F_REC_PRO_CON", "F_REC_SCORECARD"],
                       ["C_REC_VALUE_FOR_MONEY", "C_REC_STRICT_CRITERIA", "C_REC_USER_PERSONA", "C_REC_ALTERNATIVES", "C_REC_CAUTION_NOTE"]),
    "planning": (["F_PLAN_IDEA_LIST", "F_PLAN_SWOT", "F_PLAN_FRAMEWORK", "F_PLAN_MINDMAP"],
                 ["C_PLAN_FEASIBILITY", "C_PLAN_CREATIVE_MOONSHOT", "C_PLAN_KPI_MEASURABLE", "C_PLAN_STEP_BY_STEP", "C_PLAN_RISK_MITIGATION"]),
}
KEYS = ["role", "purpose", "task", "format", "constraint"]


def build_system_prompt():
    """학습 때와 같은 시스템 프롬프트 (바꾸면 모델 성능이 떨어진다)."""
    lines = [
        "사용자 질문을 분석해 AI의 role, purpose, task, format, constraint를 JSON 한 줄로 출력하세요.",
        "",
        "[선택 가능 카탈로그]",
        "task: " + ", ".join(CATALOG), # CATALOG 딕셔너리의 키값만 뽑아와 summarization, comparison... 으로 합쳐준다(구분자 쉼표)
        "",
        "format (task별, 첫 번째가 디폴트):",
    ]
    # t는 task, f는 format이다. 
    for t, (f, _) in CATALOG.items(): # 리스트로 주르륵 펼쳐진다. [(t1, t2, t3), (([f1],[c1]), ([f2],[c2]), ([f3],[c3]))]
        lines.append(f"- {t}: " + ", ".join(f)) # - summarization: format1, format2, format3 으로 합치겠다는 말
    lines += ["", "constraint (task별, 첫 번째가 디폴트):"]
    for t, (_, c) in CATALOG.items():
        lines.append(f"- {t}: " + ", ".join(c))  # - summarization: constraint1, constraint2, constraint3 으로 합치겠다는 말
    lines += [
        "",
        "[규칙]",
        "- format과 constraint는 반드시 선택한 task의 목록 안에서만 고르세요.",
        "- 질문에 형식이나 조건이 명시되지 않으면 해당 task의 디폴트를 고르세요.",
        '- 출력 키 순서: {"role", "purpose", "task", "format", "constraint"}',
    ]
    return "\n".join(lines) # ...format:, task:format...constraint:, task:constraint

SYSTEM = build_system_prompt()

FORMAT_PROMPT_CATALOG = {
    # 요약
    "F_SUM_BULLETS": "핵심 내용만 추출하여 기호(-)를 사용한 글머리 기호 형태의 문장으로 작성하세요.",
    "F_SUM_PROSE": "기호나 번호 없이 자연스럽게 이어지는 1~2개의 완결된 단락(줄글)으로 작성하세요.",
    "F_SUM_KEYWORD": "주요 핵심 개념을 **키워드**: 한 줄 설명 구조의 목록으로 작성하세요.",
    "F_SUM_TIMELINE": "시간적/논리적 전개 순서에 따라 번호 매기기(1, 2, 3...) 구조로 작성하세요.",
    # 비교
    "F_COMP_TABLE": "여러 대상과 비교 기준(속성)을 행과 열로 교차 배치한 마크다운 표([비교 기준 | 대상 A | 대상 B]) 형식으로 작성하세요.",
    "F_COMP_PRO_CON": "각 대상을 중심으로 장점과 단점 또는 특징 및 실익을 구별하여 목록화된 형태로 작성하세요.",
    "F_COMP_VENN": "공통점 영역과 핵심 차이점 영역을 명확히 분리하여 이원 구조로 작성하세요.",
    # 설명
    "F_EXP_STEPWISE": "개념과 동작 원리를 단계별(1단계, 2단계...)로 차근차근 설명하세요.",
    "F_EXP_QA": "핵심 질문(Q)과 그에 대한 명쾌한 답변(A) 구조를 반복하여 주요 의문점을 풀어주는 형태로 작성하세요.",
    "F_EXP_DEEPDIVE": "개념의 정의, 핵심 구성 요소, 실제 작동 메커니즘, 파급 효과까지 다각도로 깊이 있게 설명하세요.",
    # 추천
    "F_REC_TOPN": "가장 추천하는 항목 N개를 선정하고, 각 항목별 핵심 추천 이유와 특징을 순위/목록 형태로 정리하세요.",
    "F_REC_SITUATIONAL": "사용자의 다양한 상황, 예산, 사용 목적 등 조건별(예: '초보자용', '가성비 위주') 최적의 선택지를 분류하여 추천하세요.",
    "F_REC_PRO_CON": "추천하는 각 항목의 주요 특징과 함께 선택 시 고려해야 할 장점과 단점(주의사항)을 명확히 명시하여 추천하세요.",
    "F_REC_SCORECARD": "추천 항목들을 주요 평가 기준(가격, 성능, 편의성 등)에 따라 항목별 점수 또는 등급으로 매겨 비교 추천하세요.",
    # 기획
    "F_PLAN_IDEA_LIST": "아이디어의 명칭, 핵심 컨셉, 기대 효과를 한 세트로 묶어 3~5개의 구체적인 아이디어를 제안하세요.",
    "F_PLAN_SWOT": "기획안의 내부 요인(강점, 약점)과 외부 환경(기회, 위협)을 분석하여 전략적 방향성을 제시하세요.",
    "F_PLAN_FRAMEWORK": "문제 정의, 해결 방안, 실행 단계(Action Item), 예상 리스크 및 대응책의 구조화된 프레임워크로 작성하세요.",
    "F_PLAN_MINDMAP": "대주제를 중심으로 중주제와 세부 실행 아이디어가 가지를 치듯 확장되는 계층적 구조로 작성하세요.",
}

CONSTRAINT_PROMPT_CATALOG = {
    # 요약
    "C_SUM_LENGTH_SHORT": "전체 요약 분량을 공백 포함 300자 이내(또는 3문장 이내)로 짧고 압축적으로 작성하세요.",
    "C_SUM_LENGTH_LONG": "주요 핵심 논거와 세부 맥락이 누락되지 않도록 800자 이상의 충분한 분량으로 상세히 작성하세요.",
    "C_SUM_NO_JARGON": "전문 용어나 어려운 기술적 단어를 배제하고, 초등학생도 이해할 수 있는 쉬운 어휘로 풀어서 작성하세요.",
    "C_SUM_FACT_ONLY": "원문 분석자의 추측이나 주관적 해석을 완전히 배제하고, 원문에 명시된 객관적 수치와 팩트만 남겨 작성하세요.",
    "C_SUM_ACTION_ITEM": "전체 내용 중 향후 조치해야 할 실행 과제(Action Item) 및 후속 조치 사항 중심으로 강조하여 작성하세요.",
    # 비교
    "C_COMP_BALANCED": "어느 한쪽으로 편향되지 않도록 모든 비교 대상의 장단점과 비중을 중립적이고 균형 있게 작성하세요.",
    "C_COMP_VERDICT": "단순 비교에 그치지 않고, 각 상황이나 사용자 유형에 따라 최종적으로 어떤 대상이 더 우수한지 결론(Verdict)을 제시하세요.",
    "C_COMP_DIFF_ONLY": "공통적인 일반 기능은 생략하고, 두 대상 간의 결정적인 차이점(Differentiator) 및 핵심 차별화 요소만 집중 작성하세요.",
    "C_COMP_COST_EFFICIENCY": "성능, 가격, 유효 기간, 유지비 등 비용 대비 효과(가성비 및 ROI) 측면의 기준을 가장 최우선으로 두어 비교하세요.",
    "C_COMP_SPECIFIC_METRIC": "입력 질문에서 언급된 특정 기준(예: 보안성, 속도, 편의성 등)에 초점을 맞춰 해당 항목 위주로 깊이 있게 비교하세요.",
    # 설명
    "C_EXP_EASY_ELI5": "전문 용어를 최소화하고 쉬운 일상 어휘와 비유를 활용하여, 어린이나 초보자도 이해할 수 있도록 쉽게 작성하세요.",
    "C_EXP_TECHNICAL_DEEP": "해당 분야의 전문 용어와 정확한 메커니즘을 명확히 사용하여 전공자/전문가 수준의 깊이 있고 엄밀한 표현으로 작성하세요.",
    "C_EXP_VISUAL_ASCII": "텍스트 설명과 함께 이해를 돕기 위한 텍스트 다이어그램(ASCII 아트, 텍스트 흐름도 등)을 포함하여 시각적으로 작성하세요.",
    "C_EXP_REAL_WORLD_CASE": "추상적인 개념 설명에 그치지 않고, 실제로 적용되는 구체적인 실제 사례(Use Case)나 산업 적용 예시를 반드시 포함하세요.",
    "C_EXP_MISCONCEPTION": "해당 개념과 관련하여 사람들이 흔히 오해하거나 착각하기 쉬운 지점(Common Pitfalls)을 짚어주고 바르게 정정하여 작성하세요.",
    # 추천
    "C_REC_VALUE_FOR_MONEY": "성능, 품질, 기능 대비 가격의 합리성(가성비)을 최우선 평가 기준으로 설정하여 추천하세요.",
    "C_REC_STRICT_CRITERIA": "추천 대상 선정 시 기준이 된 분명한 평가 항목(예: 성능, 디자인, 편의성 등)과 선정이유의 객관적 근거를 명확히 제시하세요.",
    "C_REC_USER_PERSONA": "사용자의 특정 상황(예: 입문자, 독거 가구, 헤비 유저 등)이나 페르소나에 부합하는 최적의 이유를 부각하여 추천하세요.",
    "C_REC_ALTERNATIVES": "1순위 최선책(Top Pick) 외에도 대체 가능한 차선책(Alternative)이나 저렴한 옵션을 함께 명시하여 추천하세요.",
    "C_REC_CAUTION_NOTE": "단순 추천에 그치지 않고, 해당 선택 시 발생할 수 있는 리스크, 단점, 구매 전 유의사항(Caveats)을 반드시 명시하세요.",
    # 기획
    "C_PLAN_FEASIBILITY": "현실적인 예산, 인력, 기술적 제약(Feasibility)을 최우선으로 고려하여 즉시 실행 가능한 방안 위주로 작성하세요.",
    "C_PLAN_CREATIVE_MOONSHOT": "기존의 전형적인 방식이나 틀을 깨고, 파격적이고 독창적인 아이디어(Out-of-the-box)에 초점을 맞춰 제안하세요.",
    "C_PLAN_KPI_MEASURABLE": "아이디어 제안에 그치지 않고, 성공 여부를 측정할 수 있는 구체적인 정량적 성과 지표(KPI)와 목표 수치를 함께 제시하세요.",
    "C_PLAN_STEP_BY_STEP": "아이디어 기획부터 준비, 실행, 검토까지 시간 순서에 따른 단계별 로드맵(Roadmap)과 핵심 액션 아이템을 포함하세요.",
    "C_PLAN_RISK_MITIGATION": "기획안 실행 시 발생할 수 있는 예상 문제점, 법적/운영상 리스크 및 이에 대한 구체적인 대응 전략(Plan B)을 함께 명시하세요.",
}

TASK_PROMPT = {
    "summarization": "사용자가 제공한 내용을 요약하세요.",
    "comparison": "사용자가 언급한 대상들을 비교 분석하세요.",
    "explanation": "사용자가 질문한 개념을 설명하세요.",
    "recommendation": "사용자의 요청에 맞는 항목을 추천하세요.",
    "planning": "사용자의 요청에 맞는 기획안과 아이디어를 제안하세요.",
}

# 디폴트 constraint 중 일부는 중립적이지 않다 (설명=어린이 눈높이+비유, 추천=가성비 최우선).
# 모델이 디폴트를 골랐는데 질문에 해당 요청 표현이 없으면 = "조건 미지정"으로 보고 완화된 일반 지시로 바꾼다.
# 연구의 가장 큰 한계점. rule-based로 데이터를 구축한 후에 데이터에 오점이 보였으나 이미 튜닝한 이후였기 때문에 사후 처리를 하게 됨
SOFT_DEFAULT = {
    "C_EXP_EASY_ELI5": ("C_EXP_GENERAL",
                        r"쉽|초등|초보|입문|어린|아이(가|한테|에게|들|도|용|처럼)|애한테|애들|꼬마|비유|비전공|문외한|일반인|"
                        r"모르|알아듣|알아들|할머니|할아버지|부모님|노베이스|눈높이|ELI5|"
                        r"\d+\s*살|[한두세네]\s*살|다섯\s*살|여섯\s*살|일곱\s*살|조카|아들|딸(한테|에게|이)|유치원|초딩|중학생"),
    "C_REC_VALUE_FOR_MONEY": ("C_REC_GENERAL",
                              r"가성비|가심비|가격 ?대비|저렴|(?<!비)싼|(?<!비)싸(고|게|면|도)|합리적|최저가|예산|"
                              r"\d+\s*만\s*원|원 ?(이하|이내|안쪽|대)"),
}
CONSTRAINT_PROMPT_CATALOG["C_EXP_GENERAL"] = "전문 용어는 처음 나올 때 짧게 풀이하고, 핵심 개념이 명확히 전달되도록 정확하고 이해하기 쉽게 작성하세요."
CONSTRAINT_PROMPT_CATALOG["C_REC_GENERAL"] = "가격, 성능, 사용 편의성 등 주요 기준을 균형 있게 고려하여 추천하세요."

# 카탈로그 문구는 고정값(300자, 800자, N개 등)이라 사용자가 명시한 조건과 충돌할 수 있음 -> 사용자 요청 우선
PRIORITY_RULE = ("위 지침은 기본값이며, 아래 요청에 숫자(분량, 개수, 예산 등), 대상, 평가 기준, "
                 "추가 조건이 명시되어 있으면 그것을 우선하여 모두 반영하세요.")


# 라벨 후처리 : 응답 json으로부터 role, purpose, task, format, constraint를 반환하는 코드-------
def parse_label(text): # 모델 출력에서 JSON 패턴 추출
    """모델 출력 문자열 -> dict. 코드펜스나 앞뒤 잡음이 있어도 첫 JSON 객체를 꺼낸다."""
    m = re.search(r"\{.*\}", text, re.S) 
    # 텍스트 전체에서 조건에 맞는 첫 번째 객체를 꺼내주는 함수.
    # . 은 줄바꿈을 제외한 문자이기 때문에 re.S를 이용해 포함시켜준다.
    # re.search()의 결과물은 <re.match.object>이다. m.group(0)을 통해 순수 문자열을 꺼낸다.
    # json.loads는 {~~} 형태의 문자열을 딕셔너리로 반환한다. 만약{}자체가 없으면 m은 None 이므로 if else 를 통해 {}로 반환된다.
    # json.JSONDecodeError은 {A,B,} 처럼 중괄호가 있으나 그 안에 이상한 형식으로 들어가 있는 경우이다.
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


# 굳이 필요한 코드인가..?
def normalize(label, query=None): # label은 parse_label로부터 리턴값으로 받아온 딕셔너리
    """카탈로그를 벗어난 값을 보정: task -> format ID 접두사로 추론 -> 실패 시 explanation.
    format/constraint가 task 목록에 없으면 해당 task의 디폴트로 대체.
    query를 주면 SOFT_DEFAULT 규칙도 적용한다."""
    fixes = [] # 고친 결과물을 미리 선언
    task = label.get("task") # 딕셔너리로부터 'task'의 value 추출

    # 만약 task가 카탈로그에 없는 값이면
    if task not in CATALOG:
        # 번역 사전
        prefix = {"SUM": "summarization", "COMP": "comparison", "EXP": "explanation", "REC": "recommendation", "PLAN": "planning"}
        # format으로부터 언더바 기준으로 중간값을 빼온다 (ex. F_SUM_BULLETS 에서 _로 쪼개면 [F, SUM, BULLETS])
        guess = str(label.get("format", "")).split("_")[1:2]
        # 번역 사전과 guess값을 대조해서 task를 추론해낸다. 단서가 없으면 그냥 기본값이 explanation으로 반환한다.
        task = prefix.get(guess[0] if guess else "", "explanation")
        # task가 수정되었음을 표시한다. !r은 repr()을 호출하라는 뜻으로, str객체의 공식적인 문자열 표현을 반환하라는 의미이다.
        # 특수문자로 인해 버그 발생 방지를 위한 방어적 코드이다.
        fixes.append(f"task {label.get('task')!r} -> {task}")
    
    fids, cids = CATALOG[task]     # ex. 앞서 추론한 과제가 summarization인 경우 튜플로 나올 것이다.
    # 각각의 변수는 리스트를 format과 constraint에 대한 리스트를 받는다.
    # (["F_SUM_BULLETS", "F_SUM_PROSE", ...], ["C_SUM_LENGTH_SHORT", "C_SUM_LENGTH_LONG", "C_SUM_NO_JARGON",...]),
    
    # 키 값을 조회해서만약 cid가 "C_EXP_EASY_ELI5"거나 "C_REC_VALUE_FOR_MONEY"라면
    if cids[0] in SOFT_DEFAULT:  # 이미 완화된 라벨을 다시 넣어도 유지
        # cids에서 C_EXP_GENERAL나 C_REC_GENERAL를 추가하겠다는 의미
        cids = cids + [SOFT_DEFAULT[cids[0]][0]]

    # json에서 format의 밸류값이 fids안에 있다면 그것으로 채택, 아니리면 기본값
    fmt = label.get("format") if label.get("format") in fids else fids[0]
    # json에서 constraint의 밸류값이 cids안에 있다면 그것으로 채택, 아니라면 기본값
    con = label.get("constraint") if label.get("constraint") in cids else cids[0]
    # fmt에 변경이 발생했다면 그것을 fixes에 기록
    if fmt != label.get("format"):
        fixes.append(f"format {label.get('format')!r} -> {fmt}")
    # con에 변경이 발생했다면 그것을 fixes에 기록
    if con != label.get("constraint"):
        fixes.append(f"constraint {label.get('constraint')!r} -> {con}")
    # 1. 사용자의 질의가 비어있지 않고, 2. con이 SOFT_DEFAULT으로 설정되어있으며, 3. 사용자의 질의 안에 SOFT_DEFAULT의 패턴이 들어가있지 않으면
    if query is not None and con in SOFT_DEFAULT and not re.search(SOFT_DEFAULT[con][1], query):
        # 4.  C_EXP_GENERAL나 C_REC_GENERAL를 fixes의 결과로 집어넣음
        fixes.append(f"constraint {con} -> {SOFT_DEFAULT[con][0]} (요청에 해당 조건 없음)")
        # con의 결과도 바꿈. 만약 그렇지 않으면 SOFT 되기 전 버전인 C_EXP_EASY_ELI5이나 C_REC_VALUE_FOR_MONEY을 유지
        con = SOFT_DEFAULT[con][0]
    
    return {
        # 각각의 조건에 or이 붙은 경우는 모델이 role이나 purpose를 키값으로 하는 제이슨을 반환하지 않았을 수도 있기 때문임
        "role": label.get("role") or "도움이 되는 AI 어시스턴트",
        "purpose": label.get("purpose") or TASK_PROMPT[task],
        "task": task, "format": fmt, "constraint": con,
    }, fixes


# 본문과 요청을 분리하는 코드 (ex. 이거 무슨 뜻이야? 트랜스포머는........)---------------------------
# 각 줄의 시작에 있는 공백은 무시하고 # 따옴표 세 개로  열리는 지점을 찾고, 이 덩어리를 그룹1로 기억한다.
# 따옴표 뒤에 붙는 글자들을 허용하고, 해당 줄이 공백을 무시하고 끝나는 지점까지 찾고 엔터로 넘겨지는 부분을 찾음
# 열리는 지점 사이와 닫히는 지점 사이의 본문을 2로 지정한다 ?는 비탐욕적탐색 
# 본문 끝에있는 줄바꿈과 아까 닫혔던 부분을 그룹1로 찾는다 
# 닫히는 기호 뒤에 줄이 끝나는지 확인한다.
# re.M은 줄바꿈마다 ^과 &을 수행한다. re.S는 줄바꿈 포함모드이다.
# 즉 FENCE는 따옴표 계열로 열고 닫히는 본문을 캐치한다.
FENCE = re.compile(r"^\s*(\"{3}|'{3}|```)\w*\s*$\n(.*?)\n^\s*\1\s*$", re.M | re.S)

# Separtator는 === 또는 ---로 시작하고 끝나는 구분선을 찾아낸다.
SEPARATORS = re.compile(r"^\s*(?:-{3,}|={3,})\s*$", re.M)

def split_request(text, max_request=80, min_body=100):
    """요청 문장과 붙여넣은 본문(요약할 글 등)을 분리. 분류 모델에는 요청만 넣는다.
    1) 따옴표 블록(\"\"\", ''', ```)이 있으면: 블록 안 = 본문, 블록 밖 = 요청
    2) 구분선(---, ===)이 있으면: 가장 짧은 덩어리 = 요청, 나머지 = 본문
    3) 구분선 없는 여러 줄: 첫 줄/마지막 줄 중 짧은 쪽이 max_request자 이하이고
       나머지가 min_body자 이상이면서 그 2배 이상 길면 그 줄이 요청 (짧은 여러 줄 요청은 분리 안 함)
    4) 그 외: 전체가 요청, 본문 없음"""
    # 사실 프롬프트 맨 뒤에 요청: 단락을 붙이기 때문에 성능 상 차이는 없을 것을 보인다.
    # 만약 뒤에 붙은 원문이 있다면 프롬프트 맨 뒤에 \n입력 자료: 형태로 들어갈 것이다.
    # 양끝에 있는 공백과 구분자 제거
    text = text.strip()
    # FENCE.finditer(text): 정규식 패턴에 맞는 조각을 찾아 이터레이터로 반환한 m을 하나씩 순회한다.
    # m은 따옴표로 열고 닫히는 블록의 개수만큼 반환된다.
    # 정규식에서는 Match 객체를 반환한다. .group()매서드는 정규식 안에 여러 패턴이 있을 경우 유용한 매서드이다.
    # m.group(0또는 공백): m전체, m.group(n): 정규식에서 n번째 괄호로 잡아낸 내용. m.group(2)는 본문에 해당하는 부분이다.
    blocks = [m.group(2).strip() for m in FENCE.finditer(text)]
    # 블록 바깥에 있는 텍스트들
    outside = FENCE.sub("", text).strip()

    # 블록과 블록 바깥 텍스트의 분리가 성공적이라면 바깥쪽 질문과 본문을 \n\n으로 이어붙인다.
    if blocks and outside:
        return outside, "\n\n".join(blocks)
    # ---또는 ===으로 구분된 블록을 찾아낸다.
    parts = [p.strip() for p in SEPARATORS.split(text) if p.strip()]
    # ---또는 ===구분이 성공적이라면 나뉜 조간들 중에서 가장 짧은 덩어리를 질문으로 지정한다.
    if len(parts) >= 2:
        req = min(parts, key=len)
        # 똑같은 질문과 블록을  \n\n 으로 이어붙인다.
        return req, "\n\n".join(p for p in parts if p is not req)
    # 따옴표나 --- 구분선이 없다면 줄 단위로 나눈다.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # 만약 구분이 성공적이라면
    if len(lines) >= 2:
        # 첫번째 줄과 마지막 줄 중 짧은 것을 질문으로 지정한다.
        first_is_req = len(lines[0]) <= len(lines[-1])
        req = lines[0] if first_is_req else lines[-1]
        # 나머지 모든 줄들을 \n으로 모아 본문으로 지정한다.
        body = "\n".join(lines[1:] if first_is_req else lines[:-1])
        # 안전장치: 질문이 80자 이하이고 본문이 100자 이상 또는 질문의 2배이상일 때만 옳은 분리로 인정한다.
        if len(req) <= max_request and len(body) >= max(min_body, 2 * len(req)):
            return req, body
    # 앞서 모든 절차를 실패했다면 그냥 전체 텍스트를 온전히 질문으로 넘긴다.
    return text, ""

# ------------------------------------------------------------------ 프롬프트 조립 (규칙 기반 한 문단)
# 조사 붙이는 코드. 
def josa(word, a, b):
    """받침 있으면 a, 없으면 b (예: 이/가)."""
    # 마지막 글자를 가져옴 '가'는 안전장치
    c = word.rstrip()[-1:] or "가"
    # 글자가 가~힣 사이의 글자(즉 한글인지)이면서 받침이 있으면 a, 받침이 없으면b를 반환한다.
    # ord(c)는 글자를 유니코드 숫자로 변환, 0xAC00는 '가'의 유니코드, 즉 한글 시작점에서 얼마나 떨어져있는지에 대한 상대적 거리
    # 이 상대적 거리를 통해 받침 유무를 판단 가능함. 초성이 19개, 중성이 21개, 종성이 28개 이고 이 거리를 28로 나누어서 0이되면 받침이 없는 글자임
    # ex. 가(%28 = 0) 각(%28 = 1) 간(%28 = 2) 갇(%28 = 3)  갈(%28 = 4) ...
    return a if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else b

# 카탈로그로부터 키에 맞는 밸류들을 조회해서 조립하고 최정 프롬프트로 결합함
def build_prompt(query, label, content=""):
    """정규화된 라벨 -> 역할/목적/작업/형식/제약을 한 문단으로 이은 지시문 + 우선순위 규칙 + 요청 원문 (+ 입력 자료)."""
    instruction = (f"당신은 {label['role']}입니다. {label['purpose']}{josa(label['purpose'], '이', '가')} 목적이니, "
                   f"{TASK_PROMPT[label['task']]} {FORMAT_PROMPT_CATALOG[label['format']]} "
                   f"{CONSTRAINT_PROMPT_CATALOG[label['constraint']]}")
    prompt = f"{instruction} {PRIORITY_RULE}\n\n요청: {query}"
    if content:
        prompt += f"\n\n입력 자료:\n<<<\n{content}\n>>>"
    return prompt


# ------------------------------------------------------------------ 모델
class PromptOptimizer:
    """베이스 모델 + LoRA 어댑터(Hugging Face Hub 또는 로컬 경로)로 질문 -> 최적화 프롬프트."""

    def __init__(self, adapter=ADAPTER, base=BASE_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        # 의존성 주입 외부에서 torch를 쓰거나 torch 환경을 지정해야 하는 상황이 있을 때 사용함.
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base)
        # bf16 (rtx30~~)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="auto")
        # lora를 사용했으므로 PefrModel 선언
        self.model = PeftModel.from_pretrained(model, adapter).eval()

    def label(self, query):
        """질문 -> 모델이 생성한 라벨 원문 (JSON 문자열)."""
        # SYSTEM은 앞서 봤던 SYSTEM = build_system_prompt()
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            # 딕셔너리로 되어있는 enc(input_ids, attention_mask)를 언패킹해서 전달
            out = self.model.generate(**enc, max_new_tokens=160, do_sample=False)
        # 답변을 1차원으로 펴서 새롭게 생성한 답변부터 디코딩함
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, text):
        """요청(+붙여넣은 본문) -> {prompt, label, fixes, request, content}."""
        # 요청과 본문으로 분리
        request, content = split_request(text)
        # {role, purpose, format, constraint} 형태의 json과 수정사항을 반환함
        lb, fixes = normalize(parse_label(self.label(request)), request)
        # 하나의 json으로 합쳐서 반환
        return {"prompt": build_prompt(request, lb, content), "label": lb, "fixes": fixes,
                "request": request, "content": content}


# 인터페이스
def show(r):
    print("=" * 70)
    # json.dumps(): 딕셔너리나 리스트 데이터를 json형태의 문자열로 반환한다. ensure_ascii=False는 한글이 깨지지 않도록 한다.
    print("[라벨]", json.dumps(r["label"], ensure_ascii=False))
    if r["fixes"]:
        print("[보정]", "; ".join(r["fixes"]))
    print("-" * 70)
    print(r["prompt"])


if __name__ == "__main__":
    # 인수분석기. 터미널 창에 입력이 들어왔을 때 파이썬에 명령어와 함께 추가 옵션을 전달하기 위함
    
    ap = argparse.ArgumentParser()
    # 터미널 창에 문자가 들어오면 query에 싹 빠짐없이 리스트로 묶어 준다.
    ap.add_argument("query", nargs="*")  # python main.py 파이썬 리스트 컴프리헨션 알려줘
    # --adapter 옵션을 생략하면 ADAPTER를 사용하겠다는 의미이다.
    ap.add_argument("--adapter", default=ADAPTER, help="Hugging Face 저장소 ID 또는 로컬 어댑터 경로")
    # --만약 입력으로 .txt 파일이 들어오면 프로그램은 그 파일을 열어서 처리한다. python main.py --file adasda.txt
    ap.add_argument("--file", help="요청+본문이 섞인 텍스트 파일 (자동 분리)")
    # 규칙들을 하나의 보따리에 묶어준다.
    a = ap.parse_args()

    opt = PromptOptimizer(a.adapter)
    if a.file: # 만약 파일을 받았다면 ("--file"), 그 파일을 열고 옵티마이저 실행
        show(opt(open(a.file, encoding="utf-8").read()))
    elif a.query: # 만약 그냥 텍스트가 들어오면 띄어쓰기로 join하고 실행
        show(opt(" ".join(a.query)))
    else:  # 인자 없으면 대화형
        print("질문을 입력하세요 (빈 줄 입력 시 종료)")
        # 바다러스 연산자 input("> ")으로 입력값이 들어오면 여백과 공백을 씌우고 변수에 그값을 집어넣는 동시에 참인지 거짓인지 입력한다.
        # 빈줄은 거짓이므로 빈줄이 들어가면 종료되는 원리이다.
        while (q := input("> ").strip()):
            show(opt(q))
