import json, random, collections, sys

rng = random.Random(20261028)

def has_batchim(w):
    c = w[-1]
    if '가' <= c <= '힣':
        return (ord(c) - 0xAC00) % 28 != 0
    if c.isdigit():
        return c in '013678'
    return False

def jo(w, a, b):
    return w + (a if has_batchim(w) else b)

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

PREFIX = ["", "", "", "", "", "급해서 그런데 ", "내일까지 해야 돼서 ", "혹시 ", "부탁 하나만 할게. ",
          "시간이 없어서 ", "과제 때문에 ", "그냥 궁금해서 ", "저기, ", "회사에서 쓸 건데 "]

# ---------------------------------------------------------------- cues
CUES = {
 # summarization
 "F_SUM_BULLETS": ["핵심만 뽑아서", "글머리 기호로", "항목별로 끊어서"],
 "F_SUM_PROSE": ["줄글로", "자연스럽게 이어지는 문단으로", "기호 없이 글로 풀어서", "한두 단락의 글로"],
 "F_SUM_KEYWORD": ["키워드별로", "핵심 개념어 위주로 한 줄씩 설명 붙여서", "용어와 한 줄 설명 형태로", "키워드 목록으로"],
 "F_SUM_TIMELINE": ["시간 순서대로", "일어난 순서에 따라 번호 매겨서", "흐름 순서대로 1, 2, 3으로", "경과를 순서대로"],
 "C_SUM_LENGTH_SHORT": ["짧게", "간단히", "3문장 이내로", "한눈에 보이게 짧게"],
 "C_SUM_LENGTH_LONG": ["빠짐없이 자세하게", "세부 맥락까지 길고 상세하게", "논거를 놓치지 않게 충분히 길게", "디테일 살려서 상세히"],
 "C_SUM_NO_JARGON": ["어려운 용어 빼고 쉬운 말로", "전문 용어 없이 누구나 알게", "초등학생도 알아듣게 쉬운 단어로", "쉬운 말로 풀어서"],
 "C_SUM_FACT_ONLY": ["추측이나 해석 빼고 사실만", "수치와 팩트만 남겨서", "주관 없이 객관적인 사실 위주로", "원문에 나온 숫자와 사실만"],
 "C_SUM_ACTION_ITEM": ["해야 할 일 위주로", "후속 조치 사항 중심으로", "액션 아이템만 강조해서", "앞으로 실행할 과제 중심으로"],
 # comparison
 "F_COMP_TABLE": ["표로", "표 형태로", "항목별 표로"],
 "F_COMP_PRO_CON": ["각각 장점 단점으로 나눠서", "장단점 목록으로", "대상별로 장점과 단점을 구분해서", "득과 실을 나눠서"],
 "F_COMP_VENN": ["공통점과 차이점을 따로 나눠서", "겹치는 부분과 다른 부분을 구분해서", "공통점 영역과 차이점 영역으로 나눠서", "같은 점 다른 점 두 갈래로 나눠서"],
 "C_COMP_BALANCED": ["중립적으로", "한쪽으로 치우치지 않게", "공정하게"],
 "C_COMP_VERDICT": ["결국 뭐가 더 나은지 결론까지", "상황별로 어느 쪽이 우세한지 최종 판단까지 내려서", "마지막에 승자를 정해서", "누구한테 뭐가 맞는지 결론 내려서"],
 "C_COMP_DIFF_ONLY": ["차이점만", "공통 기능은 빼고 결정적인 차이만", "달라진 점만 쏙 뽑아서", "핵심 차별점 위주로"],
 "C_COMP_COST_EFFICIENCY": ["가성비 기준으로", "가격 대비 효과를 최우선으로", "유지비까지 포함한 비용 효율 중심으로", "돈값 하는지 위주로"],
 # explanation
 "F_EXP_STEPWISE": ["단계별로", "차근차근 순서대로", "1단계 2단계로"],
 "F_EXP_QA": ["Q&A 형식으로", "질문과 답변 형태로", "자주 묻는 질문 식으로", "궁금할 만한 질문 던지고 답하는 식으로"],
 "F_EXP_DEEPDIVE": ["정의부터 구성 요소, 작동 방식, 영향까지 깊이 있게", "다각도로 깊이 파고들어서", "구성과 메커니즘과 파급 효과까지 심층적으로", "하나하나 깊게"],
 "C_EXP_EASY_ELI5": ["쉽게", "초보도 알아듣게", "비유 들어서 쉽게", "어린이도 이해하게"],
 "C_EXP_TECHNICAL_DEEP": ["전문가 수준으로 용어를 정확히 써서", "전공자 눈높이로 엄밀하게", "기술적으로 정확한 용어로", "내부 메커니즘까지 학술적으로"],
 "C_EXP_VISUAL_ASCII": ["ASCII 그림으로 구조를 보여주면서", "텍스트 다이어그램을 곁들여서", "흐름도를 글자로 그려서", "도식으로 시각화해서"],
 "C_EXP_REAL_WORLD_CASE": ["실제 사례를 꼭 넣어서", "현업에서 쓰이는 예시와 함께", "산업 적용 사례 포함해서", "실무 활용 예시를 들어서"],
 "C_EXP_MISCONCEPTION": ["사람들이 흔히 오해하는 부분도 바로잡으면서", "착각하기 쉬운 점을 짚어주면서", "자주 하는 오해를 정정해주면서", "흔한 실수와 오해 위주로"],
 # recommendation
 "F_REC_TOPN": ["TOP 3로", "순위 매겨서", "베스트 5로", "랭킹으로"],
 "F_REC_SITUATIONAL": ["상황별로 나눠서", "예산대별로 분류해서", "용도별로 최적의 선택지를 나눠서", "조건별로 구분해서"],
 "F_REC_PRO_CON": ["각각 장점 단점도 같이 적어서", "항목마다 좋은 점 아쉬운 점을 붙여서", "장단점 명시해서", "특징과 함께 장단점을 정리해서"],
 "F_REC_SCORECARD": ["항목별 점수를 매겨서", "평가 기준별 등급으로", "점수표 형태로 비교하면서", "별점이나 점수로 평가해서"],
 "C_REC_VALUE_FOR_MONEY": ["가성비 좋은 걸로", "가성비 위주로", "가격 대비 괜찮은 걸로"],
 "C_REC_STRICT_CRITERIA": ["평가 기준과 근거를 분명히 밝혀서", "선정 기준이랑 객관적 근거를 대서", "어떤 기준으로 골랐는지 근거와 함께", "정량적인 근거를 제시하면서"],
 "C_REC_ALTERNATIVES": ["1순위 말고 차선책이나 대안도 같이", "플랜 B 옵션도 함께", "대체 가능한 저렴한 옵션도 같이", "1등과 대안을 같이"],
 "C_REC_CAUTION_NOTE": ["사기 전에 조심할 점도 꼭 짚어서", "리스크와 유의사항을 반드시 적어서", "단점과 주의사항 위주로 경고도 같이", "선택하면 생길 수 있는 문제점도 명시해서"],
 # planning
 "F_PLAN_IDEA_LIST": ["아이디어 몇 개", "3~5개 정도 아이디어로", "아이디어 리스트로"],
 "F_PLAN_SWOT": ["SWOT 분석으로", "강점 약점 기회 위협으로 분석해서", "SWOT 틀로 정리해서", "내부 요인과 외부 환경을 나눠서 분석하는 방식으로"],
 "F_PLAN_FRAMEWORK": ["문제 정의-해결 방안-실행 단계-리스크 구조로", "프레임워크 형태로 체계적으로", "문제 정의부터 대응책까지 구조화해서", "기획서 틀에 맞춰서"],
 "F_PLAN_MINDMAP": ["마인드맵처럼 가지를 치면서", "대주제에서 세부 아이디어로 뻗어가는 구조로", "마인드맵 형태로", "계층적으로 가지치기해서"],
 "C_PLAN_FEASIBILITY": ["현실적으로 바로 할 수 있게", "예산 적게 들고 바로 실행 가능하게", "작은 팀도 당장 할 수 있는 걸로"],
 "C_PLAN_CREATIVE_MOONSHOT": ["파격적이고 독창적으로", "기존 틀을 완전히 깨는 아이디어로", "엉뚱하고 새로운 문샷 아이디어로", "남들이 안 해본 방식으로"],
 "C_PLAN_KPI_MEASURABLE": ["KPI와 목표 수치도 같이", "성과를 측정할 수 있는 지표와 목표치를 넣어서", "정량적인 성공 기준을 제시하면서", "숫자로 평가할 수 있게 KPI까지"],
 "C_PLAN_STEP_BY_STEP": ["준비부터 실행 검토까지 단계별 로드맵으로", "시간 순서에 따른 로드맵과 액션 아이템으로", "타임라인 따라 단계별로", "일정별 로드맵 짜서"],
 "C_PLAN_RISK_MITIGATION": ["예상되는 리스크와 대응책도 함께", "문제가 생길 경우 플랜 B까지", "법적 운영상 위험요소와 대응 전략을 명시해서", "리스크 관리 관점에서"],
}

# ---- v2 augmentation: v1은 라벨당 단서 3~4개라 모델이 문구 자체를 암기함 -> 동의어/구어체 확장
MORE = {
 "F_SUM_PROSE": ["문장으로 쭉 이어서", "서술형으로", "한 편의 글처럼", "번호 없이 이어지는 글로"],
 "F_SUM_KEYWORD": ["주요 용어 정리 형식으로", "개념어마다 뜻 달아서", "핵심 단어 중심으로", "용어집처럼"],
 "F_SUM_TIMELINE": ["순서대로 번호 붙여서", "진행 과정 순으로", "타임라인으로", "어떤 순서로 진행됐는지"],
 "C_SUM_LENGTH_SHORT": ["세 줄로", "한 줄로", "짧고 굵게"],
 "C_SUM_LENGTH_LONG": ["전체 맥락 살려서 길게", "길고 꼼꼼하게", "800자 넘게 자세히", "하나도 빠짐없이"],
 "C_SUM_NO_JARGON": ["어르신도 알아들으시게", "비전공자도 이해하게 쉬운 말로", "쉬운 표현으로만", "애들도 알게"],
 "C_SUM_FACT_ONLY": ["의견은 빼고 데이터만", "팩트 위주로만", "객관적 수치만 딱", "해석 없이 사실만"],
 "C_SUM_ACTION_ITEM": ["누가 뭘 해야 하는지 위주로", "할 일 목록 중심으로", "결정 사항과 후속 업무 위주로", "투두 리스트 중심으로"],
 "F_COMP_PRO_CON": ["좋은 점 나쁜 점으로", "각각의 장단점 위주로", "이점과 단점을 따져서", "득실을 따져서"],
 "F_COMP_VENN": ["닮은 점이랑 다른 점", "공통점 차이점으로", "비슷한 점과 다른 점을 나눠서", "같은 점과 다른 점을 구분해서"],
 "C_COMP_VERDICT": ["어떤 걸 고를지 결론까지", "최종 추천까지", "그래서 뭐가 나은지 판정해서", "결론만 딱"],
 "C_COMP_DIFF_ONLY": ["바뀐 점만", "같은 건 빼고 다른 점만", "차이 나는 부분만", "다른 부분만 콕 집어서"],
 "C_COMP_COST_EFFICIENCY": ["돈 대비 어디가 이득인지", "가격 대비 성능으로", "비용 측면에서", "가성비 따져서"],
 "F_EXP_QA": ["묻고 답하는 식으로", "FAQ처럼", "문답 형식으로", "질문 답변 형태로"],
 "F_EXP_DEEPDIVE": ["심층적으로 분석해서", "원리와 영향까지 깊게", "전문 보고서처럼 깊이 있게", "구조와 파급효과까지 깊이"],
 "C_EXP_EASY_ELI5": ["쉽게 풀어서", "비전공자도 알게"],
 "C_EXP_TECHNICAL_DEEP": ["엔지니어 관점에서 엄밀하게", "논문 수준으로 정확하게", "전공 수업처럼 전문적으로", "전문 용어 써서 깊게"],
 "C_EXP_VISUAL_ASCII": ["도표나 다이어그램으로", "구조도를 텍스트로 그려서", "화살표 흐름도로", "그림으로 보여주면서"],
 "C_EXP_REAL_WORLD_CASE": ["실생활 예시 위주로", "어디에 쓰이는지 사례 중심으로", "구체적인 활용 사례와 함께", "실제 예를 들어서"],
 "C_EXP_MISCONCEPTION": ["잘못 알려진 부분을 바로잡아서", "많이들 틀리는 포인트 위주로", "오해와 진실 식으로", "헷갈리는 부분을 바로잡아서"],
 "F_REC_SITUATIONAL": ["예산별로", "상황에 따라 나눠서", "가격대별로", "목적별로"],
 "F_REC_PRO_CON": ["좋은 점 아쉬운 점 같이", "장점과 단점 하나씩 붙여서", "장단점 포함해서"],
 "F_REC_SCORECARD": ["점수 매겨서", "10점 만점으로 평가해서", "평가표로", "등급 매겨서"],
 "C_REC_STRICT_CRITERIA": ["왜 골랐는지 기준을 명확히 해서", "선정 근거를 분명히 해서", "객관적인 기준으로", "근거 확실하게"],
 "C_REC_ALTERNATIVES": ["1순위랑 대안도", "차선책까지", "안 되면 고를 다른 옵션도", "대체품도 같이"],
 "C_REC_CAUTION_NOTE": ["조심해야 할 점도 꼭", "주의사항까지", "단점이나 함정도 꼭 알려주면서", "사기 전에 체크할 위험 요소도"],
 "F_PLAN_SWOT": ["강점 약점 기회 위협 따져서", "SWOT으로 따져서", "SWOT 관점에서"],
 "F_PLAN_FRAMEWORK": ["문제부터 해결책, 실행 방안까지 체계적으로", "문제 정의와 해결 방안 중심으로 구조화해서", "체계적인 기획서 구조로", "문제-해법-실행-리스크 순으로"],
 "F_PLAN_MINDMAP": ["마인드맵으로 펼쳐서", "가지 뻗듯이", "트리 구조로", "생각을 가지치기하듯"],
 "C_PLAN_CREATIVE_MOONSHOT": ["아무도 생각 못 한 걸로", "미친 듯한 아이디어로", "틀을 깨는 방식으로", "완전 새롭게"],
 "C_PLAN_KPI_MEASURABLE": ["성과 지표랑 목표치 포함해서", "측정 가능한 목표와 함께", "KPI 꼭 넣어서", "목표 숫자까지 정해서"],
 "C_PLAN_STEP_BY_STEP": ["주차별 로드맵으로", "월별 계획으로", "단계별 실행 일정으로", "D-day 기준 일정표로"],
 "C_PLAN_RISK_MITIGATION": ["문제 될 만한 거랑 대비책도", "위험 요소와 대응 방안까지", "실패 시 대안까지", "리스크 대비책 포함해서"],
}
for k, v in MORE.items():
    CUES[k] = CUES[k] + v

METRICS = ["보안성", "속도", "배터리 시간", "사용 편의성", "소음", "무게", "확장성", "학습 난이도", "내구성", "화질"]
PERSONAS = ["자취 시작한 사회초년생", "입문자", "대학생", "초보 부모", "야근 많은 직장인", "1인 가구", "헤비 유저", "시니어", "학생", "프리랜서",
            "비전공자", "운동 초보", "신혼부부", "수험생", "캠핑 초보", "재택근무자"]

# 단서와 무관한 맥락 문장 (라벨에 영향 없음)
CONTEXT = {
 "summarization": ["자료가 너무 길어서.", "읽을 시간이 없어.", "팀장님한테 보고해야 해서.", "내용이 복잡하네."],
 "comparison": ["둘 중에 고민 중이야.", "뭘 고를지 모르겠어서.", "주변에서 의견이 갈려서.", "요즘 둘 다 많이 쓰길래."],
 "explanation": ["요즘 자주 들리는데", "뉴스에 나오길래", "수업에 나왔는데", "회의 때 나와서"],
 "recommendation": ["이번에 하나 장만하려고.", "선물하려고 하는데", "곧 필요할 것 같아서.", "고민 중이야."],
 "planning": ["다음 달에 해야 하는데", "아이디어가 안 떠올라서.", "윗선에서 지시가 내려와서.", "이번 분기 과제야."],
}

# ---------------------------------------------------------------- verbs
VERBS = {
 "summarization": (["요약해줘", "요약해봐", "간추려줘", "요약 좀", "줄여줘", "정리해줘", "핵심만 추려줘", "요점 정리해줘", "요점만 써줘", "요약 ㄱㄱ"],
                   ["요약해 주세요", "요약해 주실 수 있나요?", "간추려 주세요", "요약 부탁드립니다", "정리해 주세요", "요점만 정리해 주실래요?"]),
 "comparison": (["비교해줘", "비교해봐", "비교 좀 해줘", "차이 알려줘", "뭐가 다른지 알려줘", "차이가 뭐야?", "비교 ㄱ"],
                ["비교해 주세요", "비교해 주실 수 있나요?", "비교 부탁드립니다", "어떤 차이가 있나요?", "비교해 주실래요?"]),
 "explanation": (["설명해줘", "알려줘", "이해시켜줘", "설명해봐", "가르쳐줘", "풀어서 말해줘", "설명 ㄱ"],
                 ["설명해 주세요", "설명해 주실 수 있나요?", "알려 주세요", "설명 부탁드립니다", "가르쳐 주실래요?"]),
 "recommendation": (["추천해줘", "추천 좀", "추천해봐", "뭐가 좋은지 추천해줘", "골라줘", "뭐가 좋을까?", "추천 ㄱ"],
                    ["추천해 주세요", "추천해 주실 수 있나요?", "추천 부탁드립니다", "골라 주세요", "어떤 게 좋을까요?"]),
 "planning": (["기획해줘", "브레인스토밍 해줘", "기획안 짜줘", "아이디어 줘", "생각해봐", "계획 짜줘", "구상해줘", "방법 생각해줘"],
              ["기획해 주세요", "브레인스토밍 부탁드립니다", "기획안을 짜 주실 수 있나요?", "아이디어를 제안해 주세요", "구상해 주세요", "방안을 짜 주세요"]),
}

# ---------------------------------------------------------------- topics
SUM = [
 ("업무 문서 요약 전문가", ["어제 팀 회의록", "주간 업무 보고서", "분기 실적 보고서", "프로젝트 진행 현황 문서", "고객 불만 접수 내역", "계약서 핵심 조항", "사업계획서 초안", "경쟁사 분석 자료"]),
 ("뉴스 에디터", ["오늘 경제 뉴스 기사", "이 칼럼", "사건 사고 기사", "해외 IT 뉴스", "정부 정책 발표문", "인터뷰 기사", "부동산 시장 동향 기사"]),
 ("학술 자료 요약 전문가", ["논문 초록", "강의 노트", "이 백서", "연구 보고서", "교재 3장 내용", "학회 발표 자료", "세미나 녹취록"]),
 ("기술 문서 요약 전문가", ["API 문서", "릴리즈 노트", "장애 보고서", "기술 블로그 글", "오픈소스 README", "코드 리뷰 코멘트 모음", "시스템 설계 문서"]),
 ("콘텐츠 요약 에디터", ["이 소설 줄거리", "영화 리뷰 모음", "책 내용", "에세이", "드라마 줄거리", "여행 후기 글", "서평"]),
 ("일상 정보 요약 도우미", ["단톡방 대화 내용", "이메일 스레드", "제품 사용 설명서", "보험 약관", "유튜브 영상 내용", "아파트 관리 공지문", "병원 안내문"]),
]
COMP = [
 ("IT 기기 전문 리뷰어", [("아이폰15", "갤럭시S24"), ("맥북 에어", "그램"), ("에어팟", "갤럭시 버즈"), ("아이패드", "갤럭시 탭"), ("플레이스테이션", "닌텐도 스위치"), ("크롬북", "윈도우 노트북")]),
 ("개발 기술 컨설턴트", [("파이썬", "자바스크립트"), ("리액트", "뷰"), ("마이SQL", "몽고디비"), ("도커", "가상머신"), ("깃허브", "깃랩"), ("자바", "코틀린"), ("REST", "GraphQL")]),
 ("금융 상담 전문가", [("적금", "예금"), ("주식", "채권"), ("체크카드", "신용카드"), ("전세", "월세"), ("실비보험", "정기보험"), ("ETF", "펀드")]),
 ("생활 정보 컨설턴트", [("전기차", "하이브리드차"), ("드럼세탁기", "통돌이세탁기"), ("공기청정기", "제습기"), ("자취방", "고시원"), ("버스", "지하철"), ("에어프라이어", "전자레인지")]),
 ("학술 개념 해설가", [("귀납법", "연역법"), ("거시경제학", "미시경제학"), ("양적 연구", "질적 연구"), ("르네상스", "계몽주의"), ("바이러스", "세균"), ("자유주의", "공리주의")]),
 ("경영 컨설턴트", [("애자일", "워터폴"), ("스타트업", "대기업"), ("정규직", "프리랜서"), ("B2B", "B2C"), ("OKR", "KPI"), ("재택근무", "사무실 출근")]),
 ("콘텐츠 서비스 분석가", [("넷플릭스", "디즈니플러스"), ("유튜브 프리미엄", "스포티파이"), ("네이버", "구글"), ("카카오톡", "텔레그램"), ("쿠팡", "네이버쇼핑"), ("노션", "옵시디언")]),
]
EXP = [
 ("개발 개념 해설가", ["API", "도커", "깃 브랜치", "캐시", "해시 함수", "재귀 함수", "객체지향 프로그래밍", "데이터베이스 인덱스", "비동기 처리"]),
 ("인공지능 강사", ["트랜스포머", "머신러닝", "경사하강법", "과적합", "LLM", "파인튜닝", "RAG", "임베딩"]),
 ("경제 금융 해설가", ["인플레이션", "기준금리", "ETF", "복리", "환율", "공매도", "채권 가격과 금리의 관계"]),
 ("과학 커뮤니케이터", ["블랙홀", "광합성", "면역 체계", "기후변화", "백신 원리", "양자컴퓨터", "DNA 복제"]),
 ("생활 정보 안내자", ["연말정산", "전세 보증보험", "신용점수", "와이파이 원리", "건강보험 구조", "자동차 보험 할증"]),
 ("경영 지식 강사", ["OKR", "마케팅 퍼널", "브랜딩", "손익분기점", "린 스타트업", "제로베이스 예산"]),
 ("네트워크 보안 강사", ["DNS", "VPN", "HTTPS", "로드밸런서", "TCP 3-way 핸드셰이크", "방화벽"]),
]
REC = [
 ("IT 기기 구매 컨설턴트", ["무선 이어폰", "노트북", "보조배터리", "모니터", "기계식 키보드", "태블릿", "스마트워치"]),
 ("생활용품 큐레이터", ["로봇청소기", "에어프라이어", "운동화", "캠핑 의자", "공기청정기", "전기장판", "텀블러"]),
 ("여행 맛집 가이드", ["서울 데이트 코스", "제주도 맛집", "주말 당일치기 여행지", "국내 가족 여행지", "홈카페 원두", "밀키트", "회식 장소"]),
 ("콘텐츠 큐레이터", ["넷플릭스 드라마", "입문용 소설", "자기계발서", "보드게임", "영화", "팟캐스트", "웹툰"]),
 ("학습 자료 컨설턴트", ["프로그래밍 입문 책", "온라인 강의 사이트", "코딩 툴", "노트 앱", "영어 공부 앱", "취업용 자격증"]),
 ("금융 상품 상담사", ["적금 상품", "재테크 입문서", "신용카드", "ETF 상품", "가계부 앱", "연금 상품"]),
]
PLAN = [
 ("마케팅 디렉터", ["카페 신메뉴 홍보", "신제품 런칭 프로모션", "인스타그램 팔로워 늘리기", "SNS 이벤트", "브랜드 인지도 상승", "오프라인 팝업스토어"]),
 ("사업 기획 컨설턴트", ["사내 워크숍", "신규 사업 아이템", "직원 복지 제도", "온라인 쇼핑몰 오픈", "팀 생산성 개선", "고객 이탈 줄이기"]),
 ("프로덕트 매니저", ["앱 신규 기능", "사이드 프로젝트", "개발팀 코드 리뷰 문화", "서비스 온보딩 개선", "구독 모델 도입"]),
 ("행사 기획 전문가", ["동아리 신입 모집", "학교 축제 부스", "졸업 프로젝트 주제", "스터디 모임 운영", "송년회 이벤트"]),
 ("콘텐츠 전략가", ["유튜브 채널 콘텐츠", "블로그 주제", "팟캐스트 시작", "뉴스레터 운영", "숏폼 영상 콘텐츠"]),
 ("생활 기획 도우미", ["친구 생일 이벤트", "가족 여행 계획", "이사 준비", "부업 시작", "자기계발 루틴", "새해 목표"]),
]

SUFFIX = {"summarization": "요약", "comparison": "비교", "explanation": "개념 설명", "recommendation": "추천", "planning": "기획 아이디어 도출"}


def pick_cue(cid, default, p_default=0.35):
    if cid not in CUES or (default and rng.random() > p_default):
        return ""
    return rng.choice(CUES[cid])


def metric_cue():
    m = rng.choice(METRICS)
    return m, rng.choice([f"{m} 기준으로 집중해서", f"특히 {m} 위주로 깊이 있게", f"{m} 하나만 파고들어서", f"{m} 측면에 초점 맞춰서",
                          f"{m} 측면에서만", f"{m} 관점으로", f"딴 건 됐고 {m}만"])


def make(task, fmt, con, ambiguous=False):
    fids, cids = CATALOG[task]
    fdef, cdef = fids[0], cids[0]
    tone = rng.random() < 0.5  # True=casual
    verb = rng.choice(VERBS[task][0 if tone else 1])
    pre = rng.choice(PREFIX)
    if ambiguous:
        fc = cc = ""
    else:
        fc = pick_cue(fmt, fmt == fdef)
        cc = pick_cue(con, con == cdef, 0.25)
    extra_purpose = None
    if task == "summarization":
        role, nouns = rng.choice(SUM)
        n = rng.choice(nouns)
        subj = jo(n, "을", "를")
        topic = n
        parts = [subj, fc, cc, verb]
    elif task == "comparison":
        role, pairs = rng.choice(COMP)
        a, b = rng.choice(pairs)
        if rng.random() < 0.5:
            a, b = b, a
        subj = rng.choice([f"{jo(a,'이랑','랑')} {b}", f"{a} vs {b}", f"{jo(a,'과','와')} {b}", f"{a}하고 {b}"])
        topic = f"{jo(a,'과','와')} {b}"
        if con == "C_COMP_SPECIFIC_METRIC" and not ambiguous:
            m, cc = metric_cue()
        parts = [subj, fc, cc, verb]
    elif task == "explanation":
        role, cs = rng.choice(EXP)
        c = rng.choice(cs)
        subj = rng.choice([f"{c}에 대해", f"{jo(c,'이','가')} 뭔지", f"{jo(c,'을','를')}", f"{jo(c,'은','는')}", f"{jo(c,'을','를')} 이해하고 싶은데"])
        if subj.endswith(("은", "는")):
            subj = subj + " 대체 뭔지"
        if subj.endswith("이해하고 싶은데"):
            verb = rng.choice(["설명해줘", "알려줘"] if tone else ["설명해 주세요", "알려 주세요"])
        topic = c
        parts = [subj, fc, cc, verb]
    elif task == "recommendation":
        role, items = rng.choice(REC)
        it = rng.choice(items)
        subj = jo(it, "을", "를")
        topic = it
        persona = ""
        if con == "C_REC_USER_PERSONA" and not ambiguous:
            p = rng.choice(PERSONAS)
            persona = rng.choice([f"나 {p}인데", f"{p}인데", f"{p} 기준으로"])
            cc = ""
        if persona:
            pre = pre + persona + " "  # 페르소나는 항상 문장 앞
        parts = [subj, fc, cc, verb]
    else:
        role, tps = rng.choice(PLAN)
        t = rng.choice(tps)
        subj = rng.choice([f"{t} 관련해서", f"{jo(t,'을','를')} 위해", f"{t}"])
        topic = t
        parts = [subj, fc, cc, verb]
    # v2: 골격 다양화 - 단서 순서 섞기 / 단서를 주제 앞으로 / 맥락 문장
    *head, verb_ = parts
    lead = [p for p in head if p not in (fc, cc) or not p]
    cues = [p for p in (fc, cc) if p]
    rng.shuffle(cues)
    if cues and rng.random() < 0.25:
        body = cues + lead + [verb_]
    else:
        body = lead + cues + [verb_]
    if not pre and rng.random() < 0.3:
        pre = rng.choice(CONTEXT[task]) + " "
    q = pre + " ".join(p for p in body if p)
    if rng.random() < 0.08 and not ambiguous and "부탁" not in q:
        q = q + (" 부탁해" if tone else " 부탁드려요")
    return {"query": q.strip(), "role": role, "purpose": f"{topic} {SUFFIX[task]}", "task": task, "format": fmt, "constraint": con}


rows, seen = [], set()
for task, (fids, cids) in CATALOG.items():
    combos = [(f, c) for f in fids for c in cids]
    # explicit cue block: 160 samples, round-robin across all combos
    i = 0
    while sum(1 for r in rows if r["task"] == task) < 160:
        f, c = combos[i % len(combos)]
        i += 1
        for _ in range(50):
            r = make(task, f, c)
            if r["query"] not in seen:
                seen.add(r["query"]); rows.append(r); break
    # ambiguous block: 40 samples -> defaults
    n = 0
    while n < 40:
        r = make(task, fids[0], cids[0], ambiguous=True)
        if r["query"] not in seen:
            seen.add(r["query"]); rows.append(r); n += 1

rng.shuffle(rows)
out = sys.argv[1]
with open(out, "w", encoding="utf-8") as fp:
    for r in rows:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------------------------------ validation
errs = 0
for r in rows:
    fids, cids = CATALOG[r["task"]]
    if r["format"] not in fids or r["constraint"] not in cids:
        errs += 1
print("rows", len(rows), "invalid", errs, "unique queries", len({r['query'] for r in rows}))
print(collections.Counter(r["task"] for r in rows))
cc_ = collections.Counter((r["task"], r["format"], r["constraint"]) for r in rows)
print("combos", len(cc_), "min", min(cc_.values()), "max", max(cc_.values()))
print("default format:", sum(r["format"] == CATALOG[r["task"]][0][0] for r in rows), "default constraint:", sum(r["constraint"] == CATALOG[r["task"]][1][0] for r in rows))
print("avg query len", sum(len(r["query"]) for r in rows) / len(rows))
