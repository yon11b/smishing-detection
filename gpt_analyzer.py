"""
gpt_analyzer.py
───────────────
data dict → GPT → 최종 분석 결과 리턴

악성일 때:
{
    "is_malicious": True,
    "confidence": 0.9375,
    "sms": {
        "is_malicious": True,
        "confidence": 0.94,
        "type_distribution": { "원격제어": 0.02, ... },
        "reason": "...",
        "actions": ["...", ...]
    },
    "url": {
        "is_malicious": True,
        "confidence": 0.87,
        "reason": "...",
        "actions": ["...", ...]
    }
}

정상일 때:
{
    "is_malicious": False,
    "sms": { "is_malicious": False, "confidence": 0.12 },
    "url": { "is_malicious": False, "confidence": 0.08 }
}
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

THRESHOLD = 0.7

TYPE_GUIDE_EXAMPLES = {
    "원격제어": [
        "요청받은 원격제어 앱(TeamViewer, AnyDesk 등)을 설치하지 마세요. 이미 설치했다면 즉시 삭제하세요.",
        "앱이 설치됐다면 즉시 Wi-Fi와 모바일 데이터를 끄고 원격 세션을 종료하세요.",
        "금융앱·이메일·SNS 비밀번호를 즉시 변경하세요.",
        "한국인터넷진흥원(118) 신고 후 거래 금융기관에 연락해 계좌 이상 거래를 확인하세요.",
    ],
    "기관사칭": [
        "국가기관은 원칙적으로 URL 링크가 포함된 안내 문자를 발송하지 않습니다.",
        "문자 내 포함된 인터넷 주소(URL)는 절대 클릭하지 마시고, 해당 기관의 공식 대표번호로 직접 전화를 걸어 사실 여부를 확인하세요.",
        "피해 발생시 즉시 전기통신금융사기 통합대응센터(https://www.counterscam112.go.kr) 또는 KISA 불법스팸대응센터(spam.kisa.or.kr)에 신고하세요.",
    ],
    "결제요구": [
        "문자 내에 포함된 URL링크나 번호로 절대 연락하지 마세요.",
        "이미 링크를 클릭하여 카드 번호나 CVC 번호를 입력했다면, 즉시 해당 카드사에 분실 신고 및 승인 거절을 요청하세요.",
        "피해 발생시 즉시 전기통신금융사기 통합대응센터(https://www.counterscam112.go.kr)에 신고하세요.",
    ],
    "지인사칭": [
        "가족이나 지인의 송금 요청 문자를 받으면 반드시 기존 번호로 직접 전화해 본인 여부를 확인하세요.",
        "피해 발생시 즉시 전기통신금융사기 통합대응센터(https://www.counterscam112.go.kr)에 신고하세요.",
    ],
    "광고성": [
        "출처 불명의 이벤트·당첨 링크는 클릭하지 마세요.",
        "해당 번호를 스팸으로 신고하고 차단하세요.",
        "링크를 눌렀다면 최근 설치된 앱 목록을 확인하고 불명 앱을 삭제하세요.",
        "한국인터넷진흥원 불법스팸대응센터(118)에 신고하세요.",
    ],
}

SMS_SCHEMA = """
{
  "reason": "SMS가 악성인 이유 (2~3문장)",
  "actions": ["대응 행동1", "대응 행동2", ...]
}
"""

URL_SCHEMA = """
{
  "reason": "URL이 악성인 이유 (1~2문장)",
  "actions": ["대응 행동1", "대응 행동2", ...]
}
"""


def _get_active_types(type_distribution: dict) -> list[str]:
    active = [t for t, score in type_distribution.items() if score >= THRESHOLD]
    return active if active else [max(type_distribution, key=type_distribution.get)]


# ===============================
# SMS 분석
# ===============================
def _analyze_sms(data: dict) -> dict:
    # 정상이면 간단히 리턴
    if not data["sms_is_malicious"]:
        return {
            "is_malicious": False,
            "confidence":   data["sms_confidence"],
        }

    active_types  = _get_active_types(data["sms_type_distribution"])
    type_label    = " + ".join(active_types)
    guide_example = []
    for t in active_types:
        guide_example.extend(TYPE_GUIDE_EXAMPLES.get(t, []))

    prompt = "\n".join([
        "아래 SMS 분석 데이터를 보고 반드시 JSON으로만 응답하세요. 설명, 마크다운, 코드블록 없이 JSON만 출력하세요.",
        f"응답 스키마: {SMS_SCHEMA}",
        "",
        "[문자 내용]",
        data["sms_text"],
        "",
        "[SMS 모델 결과]",
        f"- 악성 여부: 악성 (신뢰도: {data['sms_confidence']:.1%})",
        f"- 유형 분포: {json.dumps(data['sms_type_distribution'], ensure_ascii=False)}",
        f"- 주요 유형: {type_label}{'  (복합 유형)' if len(active_types) > 1 else ''}",
        "",
        f"[{type_label} 유형 대응 가이드 예시 - 현재 문자 내용에 맞게 구체화하세요]",
        json.dumps(guide_example, ensure_ascii=False, indent=2),
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 스미싱·사이버 사기 전문 보안 분석가입니다. 반드시 한국어 JSON만 응답하세요."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    gpt_result = json.loads(response.choices[0].message.content)

    return {
        "is_malicious":      True,
        "confidence":        data["sms_confidence"],
        "type_distribution": data["sms_type_distribution"],
        "reason":            gpt_result.get("reason", ""),
        "actions":           gpt_result.get("actions", []),
    }


# ===============================
# URL 분석
# ===============================
def _analyze_url(data: dict) -> dict | None:
    if data.get("url") is None or data.get("url_is_malicious") is None:
        return None

    # 정상이면 간단히 리턴
    if not data["url_is_malicious"]:
        return {
            "is_malicious": False,
            "confidence":   data["url_confidence"],
        }

    prompt = "\n".join([
        "아래 URL 분석 데이터를 보고 반드시 JSON으로만 응답하세요. 설명, 마크다운, 코드블록 없이 JSON만 출력하세요.",
        f"응답 스키마: {URL_SCHEMA}",
        "",
        f"[URL] {data['url']}",
        f"[URL 모델 결과] 악성 (신뢰도: {data['url_confidence']:.1%})",
        f"[SMS 문자 내용 참고] {data['sms_text']}",
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 URL 보안 분석 전문가입니다. 반드시 한국어 JSON로만 응답하세요. 해당 URL이 왜 악성인지 구조, 패턴 등을 설명하세요."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    gpt_result = json.loads(response.choices[0].message.content)

    return {
        "is_malicious": True,
        "confidence":   data["url_confidence"],
        "reason":       gpt_result.get("reason", ""),
        "actions":      gpt_result.get("actions", []),
    }


# ===============================
# 메인 함수
# ===============================
def analyze(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict
        sms_text, url, sms_is_malicious, sms_confidence,
        sms_type_distribution, url_is_malicious, url_confidence

    Returns
    -------
    악성: { "is_malicious", "confidence", "sms": {...full...}, "url": {...full...} }
    정상: { "is_malicious": False, "sms": { "is_malicious", "confidence" }, "url": { ... } }
    """
    sms_result = _analyze_sms(data)
    url_result = _analyze_url(data)

    is_malicious = sms_result["is_malicious"] or bool(url_result and url_result["is_malicious"])

    # 정상이면 간단한 형식으로 리턴
    if not is_malicious:
        return {
            "is_malicious": False,
            "sms": {
                "is_malicious": False,
                "confidence":   sms_result["confidence"],
            },
            "url": {
                "is_malicious": url_result["is_malicious"] if url_result else None,
                "confidence":   url_result["confidence"]   if url_result else None,
            },
        }

    # 악성이면 전체 상세 정보 리턴
    confidences = [sms_result["confidence"]]
    if url_result and url_result.get("confidence") is not None:
        confidences.append(url_result["confidence"])
    overall_confidence = round(sum(confidences) / len(confidences), 4)

    return {
        "is_malicious": True,
        "confidence":   overall_confidence,
        "sms": {
            "is_malicious":      sms_result["is_malicious"],
            "confidence":        sms_result["confidence"],
            "type_distribution": sms_result.get("type_distribution"),
            "reason":            sms_result.get("reason"),
            "actions":           sms_result.get("actions"),
        },
        "url": {
            "is_malicious": url_result["is_malicious"] if url_result else None,
            "confidence":   url_result.get("confidence") if url_result else None,
            "reason":       url_result.get("reason")     if url_result else None,
            "actions":      url_result.get("actions")    if url_result else None,
        },
    }