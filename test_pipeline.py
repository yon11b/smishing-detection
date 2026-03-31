"""
test_pipeline.py
─────────────────
smishing_kobert.pt 없이 나머지 파이프라인 전체 테스트
pt 파일 나오면 sms_predict_dummy → sms_binary_class.predict 로 교체만 하면 됨
"""

from sms_5_class import classify_message
from url_phishing_predict import URLPhishingDetector
from gpt_analyzer import analyze as gpt_analyze

# ─────────────────────────────────────────────────────────
# SMS 모델 더미 (pt 나오면 아래 주석 해제하고 더미 함수 삭제)
# ─────────────────────────────────────────────────────────
from sms_binary_class import predict as sms_predict  # ← pt 나오면 이걸로 교체

# def sms_predict(text: str) -> dict:
#     """임시 더미 - pt 파일 나오면 삭제"""
#     return {
#         "is_malicious": True,
#         "confidence":   0.85,
#     }


# ─────────────────────────────────────────────────────────
# 파이프라인 테스트 함수
# ─────────────────────────────────────────────────────────
def test(sms_text: str, url: str = None):
    import json

    print("=" * 50)
    print(f"SMS : {sms_text}")
    print(f"URL : {url}")
    print("=" * 50)

    # 1) SMS 악성 분류
    sms_result = sms_predict(sms_text)
    print(f"\n[1] SMS 악성 분류")
    print(f"    is_malicious : {sms_result['is_malicious']}")
    print(f"    confidence   : {sms_result['confidence']}")

    # 2) SMS 유형 분류
    type_result           = classify_message(sms_text)
    sms_type_distribution = type_result["type_distribution"]
    print(f"\n[2] SMS 유형 분류")
    print(f"    primary_type : {type_result['primary_type']}")
    print(f"    distribution : {sms_type_distribution}")

    # 3) URL 탐지
    detector   = URLPhishingDetector()
    url_result = detector.predict_url(url) if url else None
    url_prob   = detector.get_url_prob(url) if url else 0.0
    print(f"\n[3] URL 탐지")
    if url_result:
        print(f"    prediction    : {url_result['prediction']}")
        print(f"    prob_phishing : {url_result['prob_phishing']}")
        print(f"    reason        : {url_result['reason']}")
    else:
        print("    URL 없음")

    # 4) 최종 위험도
    url_weight = 0.7
    sms_weight = 0.3
    risk_score = url_prob * url_weight + sms_result["confidence"] * sms_weight
    # if url:
    #     risk_score = max(url_prob, sms_result["confidence"])
    # else:
    #     risk_score = sms_result["confidence"]

    if   risk_score >= 0.6:  verdict = "악성"
    elif risk_score >= 0.35: verdict = "의심"
    else:                    verdict = "정상"

    print(f"\n[4] 최종 위험도")
    print(f"    risk_score : {round(risk_score * 100, 1)}%")
    print(f"    verdict    : {verdict}")

    # 5) GPT 분석
    url_is_malicious = None
    url_confidence   = None
    if url_result:
        url_confidence   = url_result.get("prob_phishing") or 0.0
        prediction       = url_result["prediction"]
        url_is_malicious = True if prediction == "피싱" else (False if prediction == "정상" else None)

    gpt_result = gpt_analyze({
        "sms_text":              sms_text,
        "url":                   url,
        "sms_is_malicious":      sms_result["is_malicious"],
        "sms_confidence":        sms_result["confidence"],
        "sms_type_distribution": sms_type_distribution,
        "url_is_malicious":      url_is_malicious,
        "url_confidence":        url_confidence,
    })

    print(f"\n[5] GPT 분석")
    print(json.dumps(gpt_result, ensure_ascii=False, indent=2))

    return {
        "verdict":    verdict,
        "risk_score": round(risk_score * 100, 1),
        "gpt":        gpt_result,
    }


# ─────────────────────────────────────────────────────────
# 테스트 케이스
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 케이스 1: 피싱 의심 (URL + 기관사칭)
    test(
        sms_text="고객님 계정이 해외IP에서 로그인됐습니다. 즉시 확인하세요.",
        url="www.coinonve.com",
    )

    print("\n\n")

    # 케이스 2: URL 없는 경우
    test(
        sms_text="엄마 나 폰 고장났어. 급하게 50만원만 보내줘.",
        url=None,
    )

    print("\n\n")

    # 케이스 3: 정상 메시지
    test(
        sms_text="안녕하세요. 내일 회의 10시에 시작합니다.",
        url="www.google.com",
    )

    test(
        sms_text="어제 나 밥먹고 그냥 잤어 ㅠㅠ",
        url="www.naver1.com",
    )
