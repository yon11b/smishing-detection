from __future__ import annotations

import json
import re
import time
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from phishing_pipeline import PhishingPipeline
from web_search import get_phishing_news

# 메시지 안의 URL 후보를 최대한 넓게 잡기 위한 패턴.
# http/https, www 시작 주소, 일반 도메인 형태까지 함께 추출함
BASE_DIR = Path(__file__).resolve().parent
URL_PATTERN = re.compile(
    r"""
    (?:
        https?://[^\s<>"')\]]+
        |
        www\.[^\s<>"')\]]+
        |
        (?<!@)\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b(?:/[^\s<>"')\]]*)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# 최종 판정 후 사용자에게 바로 보여줄 유형별 대응 가이드 문구.
# GPT 설명과 별개로, 빠르게 행동 지침을 안내하기 위한 고정 메시지이다.
RESPONSE_GUIDES = {
    "원격제어": """
즉시 원격 제어 앱 설치나 실행을 중단하세요.
이미 설치했다면 비밀번호를 변경하고 금융 앱과 메신저 로그인 기록을 확인하세요.
""",
    "기관사칭": """
문자 속 링크를 누르지 말고, 해당 기관의 공식 홈페이지나 대표번호로 직접 확인하세요.
납부·해지·계정 제한 같은 문구가 있어도 문자 링크로는 처리하지 않는 편이 안전합니다.
""",
    "결제요구": """
문자에 포함된 링크나 번호로 결제를 진행하지 마세요.
카드 정보를 입력했다면 카드사에 즉시 연락해 사용 정지와 이상 거래 확인을 요청하세요.
""",
    "악성 URL": """
문자에 포함된 URL은 클릭하지 말고 공식 사이트나 앱에서 직접 확인하세요.
이미 링크를 눌렀다면 비밀번호 변경, 2단계 인증 설정, 백신 검사를 진행하세요.
""",
    "지인사칭": """
송금이나 긴급 요청은 기존에 알고 있던 연락처로 다시 확인하세요.
새 번호나 메신저 대화만 믿고 송금하지 않는 편이 안전합니다.
""",
    "광고성": """
출처 불명의 이벤트·당첨 링크는 클릭하지 마세요.
해당 번호를 스팸으로 신고하고 차단하세요.
이벤트, 쿠폰, 무료 제공 문구가 있어도 개인정보 입력은 피하세요.
"""
}

# 화면 레이아웃 구성
CUSTOM_CSS = """
<style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(45, 37, 28, 0.10);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 24px rgba(60, 42, 18, 0.05);
        margin-bottom: 14px;
    }

    .stApp {
        background:
            radial-gradient(circle at top, rgba(219, 179, 107, 0.12), transparent 32%),
            linear-gradient(180deg, #f7f1e8 0%, #f8f4ec 100%);
    }

    .section-label {
        color: #8b7a67;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .result-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .result-left {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .result-title {
        color: #8b7a67;
        font-size: 12px;
    }

    .result-label {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .result-score {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 110px 1fr 56px;
        gap: 14px;
        align-items: center;
        margin-bottom: 12px;
    }

    .prob-name {
        font-size: 13px;
        color: #5d4a37;
    }

    .prob-bar {
        height: 8px;
        border-radius: 999px;
        background: #ece6dd;
        overflow: hidden;
    }

    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    .prob-value {
        text-align: right;
        font-size: 13px;
        color: #765e45;
    }

    .url-chip-wrap {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        gap: 10px;
        margin-top: 10px;
    }

    .url-chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 40px;
        min-width: 160px;
        margin: 0;
        padding: 0 12px;
        border-radius: 999px;
        background: #fff7ee;
        border: 1px solid #e5d7c9;
        color: #6a543f;
        font-size: 12px;
        box-sizing: border-box;
    }

    .url-chip-text {
        display: flex;
        align-items: center;
        height: 100%;
        margin: 0;
        line-height: 1;
        white-space: nowrap;
    }

    .url-chip-score {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 22px;
        min-width: 44px;
        margin: 0 0 0 8px;
        padding: 0 8px;
        border-radius: 999px;
        background: rgba(217, 107, 52, 0.10);
        color: #b35325;
        font-size: 11px;
        font-weight: 700;
        line-height: 1;
        white-space: nowrap;
    }

    .url-detail-grid {
        display: grid;
        gap: 12px;
        margin-top: 10px;
    }

    .url-detail-card {
        display: grid;
        gap: 12px;
        background: rgba(255, 247, 238, 0.72);
        border: 1px solid rgba(217, 107, 52, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
    }

    .url-detail-head {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        column-gap: 12px;
        margin: 0;
    }

    .url-detail-url {
        display: flex;
        align-items: center;
        height: 32px;
        margin: 0;
        font-size: 13px;
        color: #3f3024;
        line-height: 1;
        word-break: break-all;
    }

    .url-detail-score {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 32px;
        min-width: 48px;
        padding: 0 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        color: #fffaf5;
        line-height: 1;
        white-space: nowrap;
    }

    .url-detail-meta {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 0;
        align-items: stretch;
    }

    .url-meta-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 64px;
        margin: 0;
        background: rgba(255, 255, 255, 0.72);
        border-radius: 12px;
        padding: 10px 12px;
        border: 1px solid rgba(45, 37, 28, 0.08);
    }

    .url-meta-label {
        font-size: 11px;
        color: #8b7a67;
        margin: 0 0 4px 0;
    }

    .url-meta-value {
        font-size: 13px;
        font-weight: 700;
        color: #2e241a;
        margin: 0;
    }

    .url-reason {
        margin-top: 12px;
        font-size: 13px;
        color: #5b4b3c;
        line-height: 1.6;
    }

    .case-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(45, 37, 28, 0.10);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .case-date {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 999px;
        background: #f3eee7;
        color: #7c6852;
        font-size: 11px;
        margin-bottom: 8px;
    }

    .case-title {
        font-size: 14px;
        font-weight: 700;
        color: #2e241a;
        margin-bottom: 6px;
    }

    .case-summary {
        font-size: 13px;
        color: #5b4b3c;
        line-height: 1.6;
        margin-bottom: 8px;
    }

    .guide-card {
        background: #1f1d1a;
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        color: #f9f3ea;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .guide-card ul {
        margin: 10px 0 0 0;
        padding-left: 18px;
    }

    .guide-card li {
        margin-bottom: 8px;
        line-height: 1.5;
        color: #f5e7d5;
    }

    .plain-text-block {
        white-space: normal;
        word-break: break-word;
        line-height: 1.7;
        color: #4f4338;
    }

    @media (max-width: 720px) {
        .result-wrap {
            flex-direction: column;
            align-items: flex-start;
        }

        .result-score {
            font-size: 28px;
        }

        .prob-row {
            grid-template-columns: 1fr;
            gap: 6px;
        }

        .prob-value {
            text-align: left;
        }

        .url-detail-head {
            grid-template-columns: 1fr;
        }

        .url-detail-meta {
            grid-template-columns: 1fr;
        }

        .url-chip {
            width: 100%;
        }
    }
</style>
"""

# Streamlit session_state에 저장할 기본 화면 상태를 만든다.
# 분석 전/후 화면에 필요한 값들을 한 곳에서 초기화하기 위해 사용.
def make_default_state() -> dict[str, Any]:
    return {
        "bundle": None,
        "final_label": "결과",
        "final_score": 0.0,
        "sms_prob_pct": 0.0,
        "url_prob_pct": 0.0,
        "guide_text": "분석할 메시지를 입력해 주세요.",
        "ai_summary": "AI 분석 결과가 여기에 표시됩니다.",
        "latest_cases": [],
        "latest_cases_reason": None,
    }

# 분석 파이프라인은 모델과 DB를 함께 사용하므로 생성 비용이 크다.
# st.cache_resource로 한 번만 만들고 재사용해 앱 응답 속도를 높인다.
@st.cache_resource
def get_pipeline() -> PhishingPipeline:
    return PhishingPipeline(
        db_path=BASE_DIR / "phishing.db",
        model_path=BASE_DIR / "url_model_v2.pkl",
    )

# 입력 메시지에서 중복 없는 URL 목록을 추출한다.
def extract_urls(text: str) -> list[str]:
    text = (text or "").replace("\u200b", " ")

    urls: list[str] = []
    seen: set[str] = set()

    for match in URL_PATTERN.finditer(text):
        candidate = match.group(0).strip().rstrip(".,);:!?]}>\"'")

        if not candidate:
            continue

        if candidate not in seen:
            seen.add(candidate)
            urls.append(candidate)

    return urls


# SMS 본문 분석용 텍스트를 만들기 위해 메시지에서 URL만 제거한다.
# 문자 내용과 링크 자체를 분리해서 각각 다른 분석기에 넘기기 위한 단계.
def strip_message_urls(text: str, urls: list[str]) -> str:
    cleaned = text or ""
    for url in urls:
        cleaned = re.sub(re.escape(url), " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_verdict(verdict: str | None) -> str:
    mapping = {
        "피싱": "악성",
        "악성": "악성",
        "의심": "의심",
        "주의": "의심",
        "정상": "정상",
        "안전": "정상",
    }
    return mapping.get(str(verdict).strip(), "결과")


def get_final_color(label: str) -> str:
    if label == "정상":
        return "#4a9e78"
    if label == "의심":
        return "#e0a629"
    if label == "악성":
        return "#d96b34"
    return "#7f7367"


def normalize_sms_type_name(sms_type: str | None) -> str | None:
    if not sms_type:
        return None

    compact = str(sms_type).replace(" ", "")
    mapping = {
        "원격제어": "원격제어",
        "기관사칭": "기관사칭",
        "결제유도": "결제요구",
        "악성URL": "악성 URL",
        "지인사칭": "지인사칭",
        "악성 URL": "악성 URL",
        "광고성": "광고성",
        "결제요구":"결제요구"
    }
    return mapping.get(compact, str(sms_type))


def get_response_guide(final_label: str, sms_type: str | None = None) -> str:
    if final_label == "정상":
        return (
            "정상 메시지로 분류되었습니다.\n"
            "다만 출처가 불분명한 링크 클릭이나 개인정보 입력은 계속 주의하세요."
        )

    return RESPONSE_GUIDES.get(
        sms_type or "",
        "의심 또는 악성 메시지일 수 있습니다. 링크 클릭과 개인정보 입력을 멈추고 "
        "공식 경로로 다시 확인하세요.",
    )


def guess_sms_type(result: dict[str, Any]) -> str | None:
    gpt = result.get("gpt") or {}
    sms_block = gpt.get("sms") or {}
    type_distribution = sms_block.get("type_distribution") or {}
    if not type_distribution:
        return None

    try:
        sms_type = max(type_distribution, key=type_distribution.get)
    except ValueError:
        return None
    return normalize_sms_type_name(sms_type)


def safe_percent(value: Any) -> float:
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return 0.0


# UI에서 보여줄 최종 라벨은 SMS와 URL 중 더 높은 위험도를 기준으로 정한다.
# 즉, 둘 중 하나만 강하게 위험 신호를 보여도 최종 결과에 반영된다.
def decide_final_result(sms_prob_pct: float, url_prob_pct: float) -> tuple[str, float]:
    final_score = max(sms_prob_pct, url_prob_pct)

    if final_score >= 60:
        return "악성", final_score
    if final_score >= 35:
        return "의심", final_score
    return "정상", final_score

# SMS가이드 + URL 가이드를 함께 출력
def build_response_guide(bundle: dict[str, Any], primary: dict[str, Any], final_label: str) -> str:
    if final_label == "정상":
        return (
            "정상 메시지로 분류되었습니다.\n"
            "다만 출처가 불분명한 링크 클릭이나 개인정보 입력은 계속 주의하세요."
        )

    parts: list[str] = []

    sms_type = guess_sms_type(primary)
    if sms_type and sms_type in RESPONSE_GUIDES:
        parts.append("[SMS 대응 가이드]")
        parts.append(RESPONSE_GUIDES[sms_type].strip())

    has_risky_url = any(
        normalize_verdict((item.get("url_result") or {}).get("prediction")) in {"의심", "악성"}
        for item in bundle.get("results", [])
    )

    if has_risky_url and "악성 URL" in RESPONSE_GUIDES:
        parts.append("[URL 대응 가이드]")
        parts.append(RESPONSE_GUIDES["악성 URL"].strip())

    if not parts:
        return (
            "의심 또는 악성 메시지일 수 있습니다. 링크 클릭과 개인정보 입력을 멈추고 "
            "공식 경로로 다시 확인하세요."
        )

    return "\n\n".join(parts)


def build_search_query(message: str, urls: list[str]) -> str:
    cleaned_message = (message or "").strip()
    if cleaned_message:
        return cleaned_message
    if urls:
        return " ".join(urls)
    return ""

# 메시지를 실제 분석 가능한 구조로 바꾸는 함수이다.
# URL이 없으면 SMS만 분석하고, URL이 여러 개면 URL마다 결과를 만든 뒤 가장 위험한 결과를 선택해 UI의 대표 결과로 사용한다.
def analyze_message(message: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    urls = extract_urls(message)
    sms_text = strip_message_urls(message, urls)

    if not urls:
        primary = pipeline.analyze(sms_text=sms_text, url=None)
        return {
            "sms_text": sms_text,
            "urls": [],
            "results": [{"url": None, "analysis": primary, "url_result": None}],
            "primary": primary,
        }

    results = []
    for url in urls:
        analysis = pipeline.analyze(sms_text=sms_text, url=url)
        url_result = pipeline.detector.predict_url(url)
        results.append(
            {
                "url": url,
                "analysis": analysis,
                "url_result": url_result,
            }
        )

    primary = max(
        results,
        key=lambda item: float((item.get("analysis") or {}).get("risk_score", 0.0)),
    )["analysis"]

    return {
        "sms_text": sms_text,
        "urls": urls,
        "results": results,
        "primary": primary,
    }


def is_spam_or_malicious(bundle: dict[str, Any]) -> bool:
    for item in bundle.get("results", []):
        analysis = item.get("analysis") or {}
        verdict = normalize_verdict(analysis.get("verdict"))
        if verdict in {"의심", "악성"}:
            return True

        gpt = analysis.get("gpt") or {}
        sms_block = gpt.get("sms") or {}
        if sms_block.get("is_malicious"):
            return True
    return False


def parse_news_response(raw_text: str) -> list[dict[str, str]]:
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {
                    "date": str(item.get("date", "")),
                    "title": str(item.get("title", "")),
                    "summary": str(item.get("summary", "")),
                    "url": str(item.get("url", "")),
                }
                for item in data
                if isinstance(item, dict)
            ]
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [
        {
            "date": str(item.get("date", "")),
            "title": str(item.get("title", "")),
            "summary": str(item.get("summary", "")),
            "url": str(item.get("url", "")),
        }
        for item in data
        if isinstance(item, dict)
    ]

# 의심/악성으로 보이는 경우에만 최신 유사 피싱 사례를 검색한다.
# 모델 결과를 실제 국내 사례와 연결해 사용자가 더 쉽게 이해하도록 돕는다.
def search_latest_cases(message: str, bundle: dict[str, Any]) -> dict[str, Any]:
    if not is_spam_or_malicious(bundle):
        return {
            "status": "skipped",
            "items": [],
            "raw": None,
            "reason": "정상으로 분류되어 최신 사례 검색을 건너뜁니다.",
        }

    query = build_search_query(message, bundle.get("urls", []))
    if not query:
        return {
            "status": "skipped",
            "items": [],
            "raw": None,
            "reason": "검색에 사용할 메시지나 URL이 없습니다.",
        }

    try:
        sms_type = guess_sms_type(bundle.get("primary") or {})
        raw_text = get_phishing_news(query, sms_type)
        items = parse_news_response(raw_text)
        return {
            "status": "completed",
            "items": items[:3],
            "raw": raw_text,
            "reason": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "items": [],
            "raw": None,
            "reason": str(exc),
        }


# URL별 분석 결과를 카드 형태로 시각화한다.
# 판정, 위험도, 분석 단계, 사유를 함께 보여줘 링크별 위험 근거를 설명한다.
def build_ai_summary(bundle: dict[str, Any]) -> str:
    primary = bundle.get("primary") or {}
    results = bundle.get("results", [])
    gpt = primary.get("gpt") or {}
    sms_block = gpt.get("sms") or {}

    lines: list[str] = [
        f"- 최종 판정: {normalize_verdict(primary.get('verdict'))}",
        f"- 최종 위험도: {float(primary.get('risk_score', 0.0)):.1f}%",
        f"- SMS 악성확률: {safe_percent(primary.get('sms_prob', 0.0)):.1f}%",
    ]

    sms_reason = sms_block.get("reason")
    if sms_reason:
        lines.extend(["", "[SMS 분석]", str(sms_reason)])

    sms_actions = sms_block.get("actions") or []
    if sms_actions:
        lines.append("권장 조치:")
        for action in sms_actions:
            lines.append(f"- {action}")

    for index, item in enumerate(results, start=1):
        analysis = item.get("analysis") or {}
        url = item.get("url") or analysis.get("url")
        if not url:
            continue

        url_block = (analysis.get("gpt") or {}).get("url") or {}
        url_result = item.get("url_result") or {}

        lines.extend(
            [
                "",
                f"[URL 분석 {index}]",
                f"- URL: {url}",
                f"- URL 악성확률: {safe_percent(analysis.get('url_prob', 0.0)):.1f}%",
            ]
        )

        url_reason = url_block.get("reason") or url_result.get("reason")
        if url_reason:
            lines.append(f"- 사유: {url_reason}")

        url_actions = url_block.get("actions") or []
        if url_actions:
            lines.append("- 권장 조치:")
            for action in url_actions:
                lines.append(f"  - {action}")

    return "\n".join(lines)


def render_html_block(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_url_chips(results: list[dict[str, Any]]) -> None:
    urls = [item for item in results if item.get("url")]
    if not urls:
        st.caption("추출된 URL이 없습니다.")
        return

    html_parts = ['<div class="url-chip-wrap">']
    for item in urls:
        url = escape(str(item.get("url", "")))
        prob_pct = safe_percent((item.get("analysis") or {}).get("url_prob", 0.0))
        html_parts.append(
            (
                f'<div class="url-chip" title="악성확률 {prob_pct:.1f}%">'
                f'<span class="url-chip-text">{url}</span>'
                f'<span class="url-chip-score">{prob_pct:.1f}%</span>'
                f"</div>"
            )
        )
    html_parts.append("</div>")
    render_html_block("".join(html_parts))


def get_url_stage_text(stage: str | None) -> str:
    mapping = {
        "blacklist": "Blacklist",
        "ip": "IP Host",
        "shortener": "Shortener",
        "typosquatting": "Typosquatting",
        "ml": "ML",
        "invalid": "Invalid",
    }
    return mapping.get(str(stage).strip(), "Unknown")


def render_url_detail_cards(results: list[dict[str, Any]]) -> None:
    urls = [item for item in results if item.get("url")]
    if not urls:
        st.caption("URL별 분석 결과가 여기에 표시됩니다.")
        return

    html_parts = ['<div class="url-detail-grid">']

    for item in urls:
        analysis = item.get("analysis") or {}
        url_result = item.get("url_result") or {}

        raw_verdict = url_result.get("prediction") or analysis.get("verdict")
        verdict = normalize_verdict(raw_verdict)
        prob_pct = safe_percent(
            url_result.get("prob_phishing", analysis.get("url_prob", 0.0))
        )
        risk_score = safe_percent(
            url_result.get("prob_phishing", analysis.get("url_prob", 0.0))
        )
        stage = get_url_stage_text(url_result.get("stage"))
        reason = url_result.get("reason") or (
            (analysis.get("gpt") or {}).get("url") or {}
        ).get("reason", "")

        color = get_final_color(verdict)
        reason_html = ""
        if reason:
            reason_html = f'<div class="url-reason">{escape(str(reason))}</div>'

        html_parts.append(
            (
                '<div class="url-detail-card">'
                '<div class="url-detail-head">'
                f'<div class="url-detail-url">{escape(str(item.get("url", "")))}</div>'
                f'<div class="url-detail-score" style="background:{color};">{prob_pct:.1f}%</div>'
                "</div>"
                '<div class="url-detail-meta">'
                '<div class="url-meta-card">'
                '<div class="url-meta-label">판정</div>'
                f'<div class="url-meta-value">{escape(verdict)}</div>'
                "</div>"
                '<div class="url-meta-card">'
                '<div class="url-meta-label">위험도</div>'
                f'<div class="url-meta-value">{risk_score:.1f}%</div>'
                "</div>"
                '<div class="url-meta-card">'
                '<div class="url-meta-label">분석 단계</div>'
                f'<div class="url-meta-value">{escape(stage)}</div>'
                "</div>"
                "</div>"
                f"{reason_html}"
                "</div>"
            )
        )

    html_parts.append("</div>")
    render_html_block("".join(html_parts))


def render_probability_row(label: str, percent: float, color: str) -> None:
    html = f"""
    <div class="prob-row">
        <div class="prob-name">{escape(label)}</div>
        <div class="prob-bar">
            <div class="prob-bar-fill" style="width:{percent:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{percent:.1f}%</div>
    </div>
    """
    render_html_block(html)


def to_safe_html_text(text: str) -> str:
    return escape(text).replace("\n", "<br>")


def render_plain_text(text: str) -> None:
    render_html_block(f'<div class="plain-text-block">{to_safe_html_text(text)}</div>')


# 분석 결과가 생성되는 느낌을 주기 위해 줄 단위로 텍스트를 순차 출력한다.
def render_streaming_text(text: str, animate: bool) -> None:
    if not animate:
        render_plain_text(text)
        return

    placeholder = st.empty()
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(line)
        html = "<br>".join(escape(item) for item in lines)
        if hasattr(st, "html"):
            placeholder.html(f'<div class="plain-text-block">{html}</div>')
        else:
            placeholder.markdown(
                f'<div class="plain-text-block">{html}</div>',
                unsafe_allow_html=True,
            )
        time.sleep(0.03)


def create_status_box():
    if hasattr(st, "status"):
        return st.status("분석 중...", expanded=True)
    st.info("분석 중...")
    return None


def status_write(status_box, message: str) -> None:
    if status_box is not None:
        status_box.write(message)


def status_update(status_box, label: str, state: str, expanded: bool) -> None:
    if status_box is not None:
        status_box.update(label=label, state=state, expanded=expanded)


st.set_page_config(page_title="피싱/스미싱 탐지기", layout="centered")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <div style="font-size:28px; font-weight:700; color:#2d251c;">피싱/스미싱 탐지기</div>
        <div style="padding:2px 8px; border-radius:999px; background:#171717; color:#fff; font-size:11px; font-weight:700;">BETA</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-label">메시지 본문</div>', unsafe_allow_html=True)
    message = st.text_area(
        "메시지 입력",
        height=160,
        placeholder=(
            "분석할 SMS나 이메일 내용을 붙여 넣으세요.\n"
            "예) [국민은행] 고객님의 계정이 일시 정지되었습니다. 아래 링크를 확인해 주세요."
        ),
        label_visibility="collapsed",
    )
    _, button_col = st.columns([6, 1.2])
    with button_col:
        btn = st.button("분석하기", use_container_width=True)

if "analysis_state" not in st.session_state:
    st.session_state.analysis_state = make_default_state()

state = st.session_state.analysis_state

# 사용자가 분석 버튼을 누르면 여기서 전체 분석 흐름이 시작된다.
# 메시지 분석 → 최신 사례 검색 → 최종 점수 계산 → session_state 갱신 순서로 진행된다.
if btn:
    state.clear()
    state.update(make_default_state())

    if not message.strip():
        state["guide_text"] = "메시지를 입력해 주세요."
        state["ai_summary"] = "분석 결과가 없습니다."
    else:
        status_box = create_status_box()
        try:
            status_write(status_box, "1. 입력 메시지에서 URL을 추출하고 있습니다...")
            bundle = analyze_message(message)
            primary = bundle["primary"]

            status_write(status_box, "2. SMS와 URL 분석을 진행하고 있습니다...")
            news_result = search_latest_cases(message, bundle)

            sms_prob_pct = safe_percent(primary.get("sms_prob", 0.0))
            url_prob_pct = max(
                (
                    safe_percent((item.get("analysis") or {}).get("url_prob", 0.0))
                    for item in bundle["results"]
                ),
                default=0.0,
            )
            final_label, final_score = decide_final_result(sms_prob_pct, url_prob_pct)

            status_write(status_box, "3. 결과를 정리하고 있습니다...")
            state["bundle"] = bundle
            state["final_label"] = final_label
            state["final_score"] = final_score
            state["sms_prob_pct"] = sms_prob_pct
            state["url_prob_pct"] = url_prob_pct
            state["guide_text"] = build_response_guide(
                bundle, primary, final_label
            )
            state["ai_summary"] = build_ai_summary(bundle)
            state["latest_cases"] = news_result.get("items", [])
            state["latest_cases_reason"] = news_result.get("reason")

            status_update(status_box, "분석 완료", "complete", False)
        except Exception as exc:
            state["guide_text"] = "분석 중 오류가 발생했습니다."
            state["ai_summary"] = f"오류 내용: {exc}"
            state["latest_cases_reason"] = str(exc)
            status_update(status_box, "분석 중 오류 발생", "error", True)

bundle = state["bundle"]
final_label = state["final_label"]
final_score = state["final_score"]
sms_prob_pct = state["sms_prob_pct"]
url_prob_pct = state["url_prob_pct"]
guide_text = state["guide_text"]
ai_summary = state["ai_summary"]
latest_cases = state["latest_cases"]
latest_cases_reason = state["latest_cases_reason"]

animate = btn

# 아래부터는 session_state에 저장된 결과를 화면 카드들로 렌더링하는 구간이다.
# 최종 결과, 확률, URL 상세, AI 설명, 최신 사례, 대응 방법 순서로 출력된다.
with st.container(border=True):
    render_html_block(
        f"""
        <div class="result-wrap">
            <div class="result-left">
                <div class="result-title">최종 결과</div>
                <div class="result-label" style="color:{get_final_color(final_label)};">{escape(final_label)}</div>
            </div>
            <div class="result-score" style="color:{get_final_color(final_label)};">{final_score:.1f}%</div>
        </div>
        """
    )

with st.container(border=True):
    st.markdown('<div class="section-label">분류 확률</div>', unsafe_allow_html=True)
    render_probability_row("SMS 악성확률", sms_prob_pct, "#4a9e78")
    render_probability_row("URL 악성확률", url_prob_pct, "#d96b34")

with st.container(border=True):
    st.markdown('<div class="section-label">추출된 URL</div>', unsafe_allow_html=True)
    if bundle:
        render_url_chips(bundle["results"])
    else:
        st.caption("분석 후 추출된 URL이 여기에 표시됩니다.")

with st.container(border=True):
    st.markdown('<div class="section-label">URL별 분석</div>', unsafe_allow_html=True)
    if bundle:
        render_url_detail_cards(bundle["results"])
    else:
        st.caption("URL별 분석 결과가 여기에 표시됩니다.")

with st.container(border=True):
    st.markdown('<div class="section-label">피싱/스미싱 근거 - AI 분석</div>', unsafe_allow_html=True)
    render_streaming_text(ai_summary, animate)

st.markdown(
    '<div class="section-label" style="margin-top:8px;">유사 최신 피싱 여부</div>',
    unsafe_allow_html=True,
)

if latest_cases:
    for item in latest_cases:
        title = item.get("title", "제목 없음")
        date = item.get("date", "")
        summary = item.get("summary", "")
        url = item.get("url", "")

        case_html = ['<div class="case-card">']

        if date:
            case_html.append(f'<div class="case-date">{escape(date)}</div>')

        case_html.append(f'<div class="case-title">{escape(title)}</div>')

        if summary:
            case_html.append(f'<div class="case-summary">{escape(summary)}</div>')

        if url:
            case_html.append(
                f'<div class="case-summary"><a href="{escape(url)}" target="_blank">기사 보기</a></div>'
            )

        case_html.append("</div>")
        render_html_block("".join(case_html))

elif latest_cases_reason:
    render_html_block(
        f"""
        <div class="case-card">
            <div class="case-summary">{escape(str(latest_cases_reason))}</div>
        </div>
        """
    )
else:
    for index in range(3):
        render_html_block(
            f"""
            <div class="case-card">
                <div class="case-summary">최신 사례 자리 {index + 1}</div>
            </div>
            """
        )

# 대응 방법 영역은 guide_text를 줄 단위 리스트로 바꿔 카드 UI에 출력한다.
# 내부 문자열 데이터를 화면용 HTML 목록으로 변환하는 마무리 단계다.
guide_lines = [line.strip() for line in guide_text.splitlines() if line.strip()]
guide_items = "".join(f"<li>{escape(line)}</li>" for line in guide_lines) or "<li>대응 방법 정보가 없습니다.</li>"

guide_html = f"""
<div class="guide-card">
    <div style="font-size:20px; font-weight:700; margin-bottom:10px;">대응 방법</div>
    <ul>{guide_items}</ul>
</div>
"""
render_html_block(guide_html)
