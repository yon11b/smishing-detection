"""
phishing_pipeline.py
─────────────────────
모든 모델을 통합하는 메인 파이프라인

역할:
- SMS 악성 분류 모델 (KoBERT) + URL 탐지 모델 + GPT 분석을 하나로 통합
- app.py(Streamlit UI)에서 이 파일만 import해서 사용

전체 흐름:
    SMS 문자 입력
        ↓
    URL 자동 추출 + SMS 텍스트 정제 (URL 제거)
        ↓
    SMS 모델 (KoBERT) → is_malicious, confidence
    SMS 유형 분류 (룰베이스) → type_distribution
    URL 탐지 모델 → url_prob
        ↓
    최종 위험도 = max(url_prob, sms_confidence)
        ↓
    GPT 분석 → 근거 텍스트 + 대응 가이드
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from sms_binary_class import predict as sms_predict      # SMS 악성 분류 (KoBERT)
from sms_5_class import classify_message as sms_classify  # SMS 유형 분류 (룰베이스)
from url_phishing_predict import URLPhishingDetector      # URL 탐지 모델
from gpt_analyzer import analyze as gpt_analyze           # GPT 상세 분석


# ─────────────────────────────────────────────────────────
# SMS에서 URL 추출 + SMS 텍스트 정제
# ─────────────────────────────────────────────────────────
def extract_url_from_sms(sms_text: str) -> tuple[str, Optional[str]]:
    """
    SMS 문자에서 URL을 추출하고, URL이 제거된 순수 SMS 텍스트를 반환

    [왜 URL을 제거하는가?]
    KoBERT SMS 모델에 URL이 포함된 텍스트를 그대로 넣으면
    모델이 URL 패턴을 피싱 특징으로 오인해서 정상 URL도 악성으로 판단하는 문제 발생
    예) "www.naver.com" → KoBERT가 81.5% 피싱으로 오판
    → URL을 제거한 순수 텍스트만 SMS 모델에 넘겨야 정확한 판단 가능

    Returns
    -------
    (cleaned_sms_text, extracted_url)
    - cleaned_sms_text : URL이 제거된 순수 SMS 텍스트
    - extracted_url    : 추출된 첫 번째 URL (없으면 None)
    """
    # URL 패턴: http(s)://, www., 또는 알려진 TLD로 끝나는 도메인
    pattern = r'https?://\S+|www\.\S+|[A-Za-z0-9.-]+\.(?:com|co\.kr|kr|net|org|xyz|ml|tk)\S*'

    matches = re.findall(pattern, sms_text, flags=re.IGNORECASE)
    url = matches[0] if matches else None  # 첫 번째 URL만 사용

    # SMS 텍스트에서 URL 제거 후 공백 정리
    cleaned_text = re.sub(pattern, '', sms_text, flags=re.IGNORECASE).strip()

    return cleaned_text, url


# ─────────────────────────────────────────────────────────
# 메인 파이프라인 클래스
# ─────────────────────────────────────────────────────────
class PhishingPipeline:
    """
    SMS + URL 통합 피싱 탐지 파이프라인

    Streamlit app.py에서 이 클래스 하나만 import해서 사용
    내부적으로 SMS 모델, URL 모델, GPT 분석을 모두 처리
    """

    def __init__(
        self,
        db_path: Path | str = "phishing.db",
        model_path: Path | str = "url_model_v2.pkl",
        url_threshold: float = 0.60,
    ) -> None:
        """
        Parameters
        ----------
        db_path       : KISA 블랙리스트 SQLite DB 경로
        model_path    : 학습된 URL 탐지 모델 pkl 경로
        url_threshold : URL ML 모델 판정 임계값 (기본 0.60)
        """
        # URLPhishingDetector 초기화 (모델 자동 로드 포함)
        self.detector = URLPhishingDetector(
            db_path=db_path,
            model_path=model_path,
            threshold=url_threshold,
        )

    def _get_url_for_gpt(
        self, url: Optional[str]
    ) -> tuple[Optional[bool], Optional[float]]:
        """
        URL 탐지 결과를 gpt_analyzer가 받는 형식으로 변환

        gpt_analyzer는 url_is_malicious(bool)와 url_confidence(float)를 받음
        ML 모델의 "의심" 판정은 None으로 전달 (GPT가 알아서 처리)

        Returns
        -------
        (url_is_malicious, url_confidence)
        - 피싱: (True, prob_phishing)
        - 정상: (False, prob_phishing)
        - 의심: (None, prob_phishing)
        - URL 없음: (None, None)
        """
        if not url:
            return None, None

        result        = self.detector.predict_url(url)
        prediction    = result["prediction"]
        prob_phishing = result.get("prob_phishing") or 0.0

        if prediction == "피싱":
            return True,  round(prob_phishing, 4)
        elif prediction == "정상":
            return False, round(prob_phishing, 4)
        else:
            # 의심 → None으로 전달 (gpt_analyzer가 confidence 참고해서 판단)
            return None, round(prob_phishing, 4)

    def analyze(
        self,
        sms_text: str,
        url: Optional[str] = None,
    ) -> dict:
        """
        SMS 문자 + URL을 분석하여 최종 피싱 판정 결과 반환

        Parameters
        ----------
        sms_text : SMS 원문 전체 (URL 포함 가능)
        url      : 별도로 지정할 URL (None이면 sms_text에서 자동 추출)

        Returns
        -------
        {
            "verdict":    "악성" | "의심" | "정상",
            "risk_score": float (0~100, 퍼센트),
            "pct_danger": float (위험 확률 퍼센트),
            "pct_normal": float (정상 확률 퍼센트),
            "url_prob":   float (URL 피싱 확률 0~1),
            "sms_prob":   float (SMS 피싱 확률 0~1),
            "url":        str | None (탐지된 URL),
            "gpt": {
                "is_malicious": bool,
                "confidence":   float,
                "sms": { "is_malicious", "confidence", "reason", "actions" },
                "url": { "is_malicious", "confidence", "reason" } | None,
            }
        }
        """
        # ── URL 추출 + SMS 텍스트 정제 ───────────────────
        # url이 별도로 지정되지 않은 경우 SMS에서 자동 추출
        if url is None:
            # extract_url_from_sms: URL 제거된 텍스트 + 추출된 URL 반환
            sms_text, url = extract_url_from_sms(sms_text)
            # 이후 sms_text는 URL이 제거된 순수 텍스트로 사용됨

        # ── 1) SMS 악성 분류 (KoBERT) ────────────────────
        # sms_text: URL이 이미 제거된 순수 텍스트
        # KoBERT가 URL 패턴에 혼동하지 않도록 URL 제거 후 넘김
        sms_result       = sms_predict(sms_text)
        sms_is_malicious = sms_result["is_malicious"]  # True/False
        sms_confidence   = sms_result["confidence"]    # 피싱 확률 (0~1)

        # ── 2) SMS 유형 분류 (룰베이스) ──────────────────
        # 원문 그대로 사용 (유형 분류는 키워드 매칭이라 URL 있어도 무방)
        type_result           = sms_classify(sms_text)
        sms_type_distribution = type_result["type_distribution"]
        # 예: {"원격제어": 0.0, "기관사칭": 1.0, "결제요구": 0.0, ...}

        # ── 3) URL 탐지 ───────────────────────────────────
        url_prob                     = self.detector.get_url_prob(url)      # 피싱 확률 (0~1)
        url_is_malicious, url_confidence = self._get_url_for_gpt(url)      # GPT 입력용 변환

        # ── 4) 최종 위험도 계산 ───────────────────────────
        # 전략: SMS/URL 중 더 높은 확률을 최종 위험도로 사용
        # 이유: URL이 피싱이면 SMS가 정상이어도 악성으로 판단해야 하고
        #       SMS가 피싱이면 URL 없어도 위험으로 판단해야 함
        if url:
            # URL 있으면 둘 중 더 높은 확률 사용
            risk_score = max(url_prob, sms_confidence)
        else:
            # URL 없으면 SMS 확률만으로 판단
            risk_score = sms_confidence

        # 임계값 기반 최종 판정
        if   risk_score >= 0.6:  verdict = "악성"
        elif risk_score >= 0.35: verdict = "의심"
        else:                    verdict = "정상"

        # ── 5) GPT 상세 분석 ─────────────────────────────
        # SMS/URL 결과를 GPT에 넘겨서 한국어 판단 근거 + 대응 가이드 생성
        gpt_result = gpt_analyze({
            "sms_text":              sms_text,             # SMS 원문
            "url":                   url,                  # 탐지된 URL
            "sms_is_malicious":      sms_is_malicious,     # SMS 악성 여부
            "sms_confidence":        sms_confidence,       # SMS 피싱 확률
            "sms_type_distribution": sms_type_distribution, # SMS 유형 분포
            "url_is_malicious":      url_is_malicious,     # URL 악성 여부
            "url_confidence":        url_confidence,       # URL 피싱 확률
        })

        return {
            "verdict":    verdict,
            "risk_score": round(risk_score * 100, 1),        # 0~100 퍼센트
            "pct_danger": round(risk_score * 100, 1),        # 위험 확률 %
            "pct_normal": round((1 - risk_score) * 100, 1),  # 정상 확률 %
            "url_prob":   round(url_prob, 4),                # URL 피싱 확률
            "sms_prob":   round(sms_confidence, 4),          # SMS 피싱 확률
            "url":        url,                               # 탐지된 URL
            "gpt":        gpt_result,                        # GPT 분석 결과
        }


# ─────────────────────────────────────────────────────────
# 직접 실행 시 테스트
# python phishing_pipeline.py 로 실행
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    pipeline = PhishingPipeline()

    result = pipeline.analyze(
        sms_text="안녕하세요 www.naveer.com",  # 테스트 입력
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
