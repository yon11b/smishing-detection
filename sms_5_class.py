# sms_5_class.py

import re

# ===============================
# 1. 텍스트 전처리
# ===============================
def clean_text(text):
    text = str(text)
    text = text.replace('\n', ' ')
    text = re.sub(
        r'(http[s]?://\S+|www\.\S+|[A-Za-z0-9.-]+\.(?:com|co\.kr|kr|net|org|me|cc|tv)\S*)',
        ' URL ',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\b01[0-9]-?\d{3,4}-?\d{4}\b', ' PHONE ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ===============================
# 2. 유형별 패턴 사전
# ===============================
type_patterns = {
    '원격제어형': [
        r'원격',
        r'원격\s*지원',
        r'보안\s*앱',
        r'앱\s*설치',
        r'어플\s*설치',
        r'앱을?\s*깔',
        r'설치\s*후\s*실행',
        r'팀뷰어|teamviewer',
        r'애니데스크|anydesk',
        r'apk',
        r'화면\s*공유'
    ],
    '기관사칭형': [
        r'검찰|경찰|법원|금감원|금융감독원|국세청|정부24',
        r'국민은행|신한은행|우리은행|하나은행|농협|NH농협|기업은행|부산은행|새마을금고',
        r'카카오뱅크|토스|케이뱅크',
        r'우체국|한진택배|CJ대한통운|롯데택배|쿠팡',
        r'코인원|업비트|빗썸',
        r'고객님|회원님'
    ],
    '결제 요구형': [
        r'입금|송금|이체|결제|승인|취소|환불|출금|충전',
        r'수수료|정산|잔액|미납|납부',
        r'계좌|계좌번호|카드|카드번호',
        r'\d+\s*원',
        r'\d+\s*만원',
        r'\d+\s*천원'
    ],
    '지인사칭형': [
        r'엄마|아빠|누나|언니|오빠|형|동생|친구|삼촌|이모|고모|아들|딸',
        r'폰\s*고장|휴대폰\s*고장|핸드폰\s*고장',
        r'카톡\s*안|문자\s*부탁|급한\s*일|지금\s*바빠',
        r'내\s*명의|내\s*번호'
    ],
    '광고성': [
        r'무료|할인|이벤트|특가|쿠폰|혜택|광고|홍보',
        r'대출|한도|승인|상담|가입|회원가입',
        r'수익|부업|알바|주식|투자',
        r'카지노|토토|배팅|스포츠',
        r'당첨|지원금|프로모션'
    ]
}

TYPE_KEY_MAP = {
    '원격제어형': '원격제어',
    '기관사칭형': '기관사칭',
    '결제 요구형': '결제요구',
    '지인사칭형': '지인사칭',
    '광고성':     '광고성',
}


# ===============================
# 3. 패턴 매칭 함수 (내부용)
# ===============================
def _count_pattern_matches(text, patterns):
    total = 0
    for p in patterns:
        total += len(re.findall(p, text, flags=re.IGNORECASE))
    return total


# ===============================
# 4. 메인 함수 (외부에서 import해서 사용)
# ===============================
def classify_message(message: str) -> dict:
    """
    SMS 텍스트를 받아 유형 분류 결과를 반환합니다.

    Returns
    -------
    {
        "primary_type":      str,        # 대표 유형 (정상이면 '정상')
        "matched_types":     list[str],  # 매칭된 유형 목록
        "type_distribution": dict,       # 유형별 정규화 점수 (0~1)
        "spam_label":        int,        # 1: 스팸, 0: 정상
    }
    """
    content_clean = clean_text(message)

    scores = {
        type_name: _count_pattern_matches(content_clean, patterns)
        for type_name, patterns in type_patterns.items()
    }

    total = sum(scores.values())
    if total > 0:
        type_distribution = {
            TYPE_KEY_MAP[k]: round(v / total, 4)
            for k, v in scores.items()
        }
    else:
        type_distribution = {v: 0 for v in TYPE_KEY_MAP.values()}

    max_score = max(scores.values())

    if max_score == 0:
        return {
            "primary_type":      "정상",
            "matched_types":     [],
            "type_distribution": type_distribution,
            "spam_label":        0,
        }

    matched_types = [k for k, v in scores.items() if v > 0]
    primary_type  = next(k for k in type_patterns if scores[k] == max_score)

    return {
        "primary_type":      primary_type,
        "matched_types":     matched_types,
        "type_distribution": type_distribution,
        "spam_label":        1,
    }