"""
url_phishing_common.py
───────────────────────
URL 피싱 탐지에 필요한 공통 함수, 상수, 유틸리티 모음
다른 모듈(train, predict, pipeline)에서 import해서 사용
"""

from __future__ import annotations  # 파이썬 3.10 미만에서도 최신 타입 힌트 문법 사용 가능

from dataclasses import dataclass    # @dataclass 데코레이터 (자동 __init__ 생성)
from difflib import SequenceMatcher  # 두 문자열 유사도 계산 (타이포스쿼팅 감지에 사용)
from pathlib import Path             # 파일 경로를 객체로 다루는 라이브러리
import re                            # 정규표현식
from typing import Any               # 타입 힌트용
from urllib.parse import urlparse    # URL을 host, path, query 등으로 분해


# ─────────────────────────────────────────────────────────
# 기본 경로 및 설정 상수
# ─────────────────────────────────────────────────────────

# 현재 파일(url_phishing_common.py)이 위치한 폴더 경로
# __file__ = 현재 파일 자체, .resolve() = 절대경로, .parent = 상위 폴더
MODULE_DIR = Path(__file__).resolve().parent

# 각 파일의 기본 경로 (같은 폴더 안에 있다고 가정)
DEFAULT_DB_PATH            = MODULE_DIR / "phishing.db"                        # KISA 블랙리스트 SQLite DB
DEFAULT_MODEL_PATH         = MODULE_DIR / "url_model_v2.pkl"                   # 학습된 ML 모델
DEFAULT_PHISH_DATASET_PATH = MODULE_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"  # 피싱 학습 데이터
DEFAULT_TRANCO_PATH        = MODULE_DIR / "tranco.csv"                          # 정상 도메인 데이터

# ML 판정 임계값
# 피싱 확률 >= 0.60 → 피싱
# 0.35 <= 피싱 확률 < 0.60 → 의심
# 피싱 확률 < 0.35 → 정상
DEFAULT_THRESHOLD = 0.60

# 재현성을 위한 랜덤 시드 (같은 값이면 항상 동일한 결과 보장)
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────
# 단축 URL 도메인 목록
# ─────────────────────────────────────────────────────────
# set으로 만든 이유: "host in SHORTENER_DOMAINS" 조회가 list보다 훨씬 빠름 (O(1) vs O(n))
SHORTENER_DOMAINS = {
    "adf.ly", "bit.ly", "buff.ly", "cutt.ly", "goo.gl",
    "is.gd", "kr.pe", "me2.do", "ow.ly", "rb.gy",
    "t.co", "tiny.cc", "tinyurl.com", "url.kr", "vo.la",
}


# ─────────────────────────────────────────────────────────
# 위험 TLD (Top Level Domain) 목록
# ─────────────────────────────────────────────────────────
# 피싱 사이트가 자주 사용하는 TLD 목록
# 초기 16개에서 new_dataset.csv 미탐 분석 결과를 보고 14개 추가 (총 30개)
RISKY_TLDS = {
    # 기존 목록 (초기 설계 시 선정)
    "buzz", "cc", "cf", "click", "country",
    "fit", "gq", "icu", "mba", "ml",
    "rest", "shop", "tk", "top", "work", "xyz",
    # 미탐 분석 후 추가된 TLD
    # (피싱인데 정상으로 오판된 URL들의 TLD를 분석해서 추가)
    "pet", "beauty", "cash", "hair", "today",
    "biz", "black", "media", "skin", "ink",
    "gay", "agency", "uno", "live",
}


# ─────────────────────────────────────────────────────────
# 피싱 의심 키워드 목록
# ─────────────────────────────────────────────────────────
# 피싱 사이트가 신뢰감을 주려고 URL에 자주 포함시키는 단어들
# URL에 이 단어가 포함될수록 피싱 가능성이 높음
SUSPICIOUS_WORDS = [
    "account", "auth", "bank", "bonus", "claim",
    "confirm", "gift", "login", "password", "payment",
    "recovery", "secure", "signin", "support", "update",
    "verify", "wallet",
]


# ─────────────────────────────────────────────────────────
# 타이포스쿼팅 감지 대상 브랜드 목록
# ─────────────────────────────────────────────────────────
# 피싱 사이트가 사칭하는 한국 주요 금융/기관/플랫폼 브랜드 37개
# 타이포스쿼팅 예시: coinone → coinonve.com (한 글자 바꿔서 사칭)
LEGIT_BRANDS = {
    # 암호화폐 거래소
    "coinone", "upbit", "bithumb", "korbit", "gopax",
    # 은행
    "kbstar", "shinhan", "wooribank", "hanabank", "ibk",
    "nonghyup", "kakaobank", "tossbank",
    # 간편결제/핀테크
    "toss", "kakaopay", "naverpay",
    # 포털/SNS
    "naver", "kakao", "daum", "google", "facebook",
    # 공공기관
    "kisa", "fss", "nts", "police",
    # 쇼핑/배송
    "coupang", "11st", "gmarket", "auction",
}

# 타이포스쿼팅 판정 유사도 기준값
# 0.75 이상이면 타이포스쿼팅으로 판정
# 예: coinonve ↔ coinone = 약 0.93 → 타이포스쿼팅
TYPOSQUATTING_THRESHOLD = 0.75


# ─────────────────────────────────────────────────────────
# 타이포스쿼팅 감지 함수
# ─────────────────────────────────────────────────────────
def is_typosquatting(host: str) -> bool:
    """
    host가 유명 브랜드를 사칭한 타이포스쿼팅 도메인인지 판별

    동작 방식:
        1단계: 정확히 일치하는 브랜드인지 먼저 확인 (오탐 방지용)
        2단계: SequenceMatcher로 유사도 계산, 0.75 이상이면 타이포스쿼팅

    [중요] 정확 일치를 먼저 체크하는 이유:
        LEGIT_BRANDS가 set이라 순서가 랜덤함
        "naver"를 비교할 때 "naverpay"가 먼저 순회되면
        유사도가 0.77로 계산되어 naver.com을 타이포스쿼팅으로 오판할 수 있음
        → 정확 일치 체크를 먼저 해서 이 문제를 방지
    """
    parts = host.split(".")
    host_main = parts[0] if parts else host  # "naver.com" → "naver"

    # 1단계: 정확히 일치하는 브랜드가 있으면 정상 처리 (오탐 방지)
    if host_main in LEGIT_BRANDS:
        return False

    # 2단계: 유사도 비교
    # SequenceMatcher(None, a, b).ratio() → 두 문자열이 얼마나 비슷한지 0~1로 반환
    for brand in LEGIT_BRANDS:
        ratio = SequenceMatcher(None, host_main, brand).ratio()
        if ratio >= TYPOSQUATTING_THRESHOLD:
            return True  # 유사도 0.75 이상 → 타이포스쿼팅

    return False


# ─────────────────────────────────────────────────────────
# 정상 URL 생성에 사용할 경로 패턴
# ─────────────────────────────────────────────────────────
# 학습 데이터의 정상 URL을 만들 때 Tranco 도메인에 이 경로들을 조합
# 예) "naver.com" + "/about" → "https://naver.com/about"
LEGIT_PATHS = [
    "", "/", "/about", "/blog", "/category/sports",
    "/contact", "/en/index", "/faq", "/help", "/home",
    "/news", "/products", "/search?q=hello", "/services", "/shop",
]


# ─────────────────────────────────────────────────────────
# 수치 피처 컬럼 순서 (20개)
# ─────────────────────────────────────────────────────────
# StandardScaler와 ML 모델이 항상 같은 순서로 피처를 받도록 순서 고정
# 학습 시와 예측 시 순서가 다르면 결과가 틀림
NUMERIC_FEATURE_COLUMNS = [
    "is_ip", "host_len", "url_len", "path_len", "query_len",
    "dot_count", "hyphen_count", "hyphen_in_host", "slash_count",
    "digit_count", "digit_ratio", "at_count", "subdomain_count",
    "host_main_len", "vowel_count", "has_shortener", "risky_tld",
    "suspicious_word_count", "encoded_ratio", "shortener_style_score",
]


# ─────────────────────────────────────────────────────────
# URL 전처리 함수
# ─────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    """
    URL 정규화: 소문자 변환 + 앞뒤 공백 제거 + 후행 슬래시 제거

    url or "" : url이 None일 때 빈 문자열로 대체
    예) "HTTPS://Naver.com/  " → "https://naver.com"
        "naver.com/"          → "naver.com"
    """
    return str(url or "").strip().lower().rstrip("/")


def parse_parts(url: str) -> tuple[str, str, str, str]:
    """
    URL을 4개 구성요소로 분해하여 반환: (raw, host, path, query)

    예) "www.naver.com/search?q=hello" 입력 시
        raw   = "www.naver.com/search?q=hello"  (정규화된 원본)
        host  = "naver.com"                      (www. 제거됨)
        path  = "/search"
        query = "q=hello"

    [참고] urlparse가 정상 동작하려면 http:// 스킴이 필요
        스킴이 없으면 host를 path로 잘못 파싱하므로 없으면 앞에 붙여줌
    """
    raw = normalize_url(url)
    if not raw:
        return "", "", "", ""

    # urlparse 정상 동작을 위해 스킴(http://) 추가
    raw_for_parse = raw if raw.startswith(("http://", "https://")) else f"http://{raw}"
    parsed = urlparse(raw_for_parse)

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]  # www. 제거 (naver.com과 www.naver.com을 동일하게 처리)

    return raw, host, parsed.path.lower(), parsed.query.lower()


def normalize_domain(url_or_domain: str) -> str:
    """
    URL 또는 도메인에서 host 부분만 추출
    블랙리스트 DB 저장/조회 시 도메인 형태를 통일하기 위해 사용

    예) "http://www.evil.com/login" → "evil.com"
        "evil.com"                  → "evil.com"
    """
    _, host, _, _ = parse_parts(url_or_domain)
    return host.rstrip("/")


def get_tld(host: str) -> str:
    """
    host에서 TLD(최상위 도메인) 추출

    예) "naver.com" → "com"
        "evil.xyz"  → "xyz"
        "a.b.co.kr" → "kr"
    """
    parts = host.split(".")
    return parts[-1] if len(parts) >= 2 else ""


def is_ip(host: str) -> int:
    """
    host가 IP 주소인지 판별 (int 반환: 0 또는 1)

    정규식으로 0~255 범위의 숫자 4개가 .으로 연결된 IPv4 형태를 체크
    예) "192.168.1.1" → 1 (IP 주소)
        "naver.com"   → 0 (일반 도메인)

    IP 주소 host는 도메인 없이 직접 접속하는 피싱에서 자주 사용됨
    """
    pattern = r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
    return 1 if re.fullmatch(pattern, host) else 0


def subdomain_count(host: str) -> int:
    """
    서브도메인 깊이 계산

    예) "login.verify.naver.com" → 점으로 나누면 4개 → 4-2 = 2개 서브도메인
        "naver.com"               → 2개 → 2-2 = 0
        IP 주소면 0 반환

    피싱 URL은 서브도메인을 여러 개 쌓아서 정상 도메인처럼 보이게 하는 수법 사용
    예) "login.kbstar.account.evil.com"
    """
    if is_ip(host):
        return 0
    return max(0, len(host.split(".")) - 2)


def suspicious_word_count(text: str) -> int:
    """
    URL에 피싱 의심 키워드가 몇 개 포함됐는지 카운트

    예) "http://secure-login.evil.com/verify"
        → "secure", "login", "verify" 포함 → 3 반환

    피싱 사이트들은 신뢰감을 주기 위해 URL에 이런 단어를 넣는 경향이 있음
    """
    lowered = text.lower()
    return sum(1 for word in SUSPICIOUS_WORDS if word in lowered)


def shortener_style_score(host: str, path: str) -> int:
    """
    단축 URL처럼 보이는 패턴에 점수 부여 (0~4점)

    단축 URL 특징:
    - 호스트명이 짧음 (rb.gy, t.co 등)
    - 경로가 짧고 랜덤한 영숫자 조합 (예: /3xAb12)

    각 조건 당 1점씩 (총 최대 4점):
    1. 호스트 메인 파트 길이 4 이하
    2. 경로 토큰 길이 2~8
    3. 경로가 영숫자+특수문자만으로 구성
    4. 경로에 문자와 숫자가 혼합
    """
    score = 0
    parts = host.split(".")
    host_main = parts[0] if parts else host
    path_token = path.strip("/")  # 앞뒤 슬래시 제거 (예: "/3xAb12" → "3xAb12")

    if len(host_main) <= 4:
        score += 1  # 예: "rb", "t", "bit" → 짧은 호스트
    if 2 <= len(path_token) <= 8:
        score += 1  # 예: "3xAb12" → 짧은 경로
    if path_token and re.fullmatch(r"[a-zA-Z0-9_-]+", path_token):
        score += 1  # 영숫자만으로 구성된 경로
    if re.search(r"[a-zA-Z]", path_token) and re.search(r"\d", path_token):
        score += 1  # 문자 + 숫자 혼합 (예: "3xAb12")

    return score


# ─────────────────────────────────────────────────────────
# 핵심 함수: 수치 피처 추출 (20개)
# ─────────────────────────────────────────────────────────
def extract_features(url: str) -> dict[str, float]:
    """
    URL 하나에서 수치 피처 20개를 추출하여 딕셔너리로 반환

    ML 모델의 입력값이 되는 핵심 특성공학 함수
    모든 값을 float으로 변환하는 이유: StandardScaler가 float 타입을 요구함

    피처 설계 기준: 피싱 URL에서 자주 나타나는 구조적 특징을 사람이 직접 설계
    (TF-IDF가 자동으로 패턴을 찾는 것과 달리 도메인 지식을 반영한 수동 피처)
    """
    raw, host, path, query = parse_parts(url)
    tld = get_tld(host)
    parts = host.split(".")
    host_main = parts[0] if parts else host   # 도메인 메인 파트 (예: "naver")
    path_and_query = path + query
    encoded_ratio = path_and_query.count("%") / max(len(path_and_query), 1)  # URL 인코딩 문자 비율
    digit_ratio = sum(c.isdigit() for c in raw) / max(len(raw), 1)           # 숫자 문자 비율

    return {
        # 구조 특성
        "is_ip":                 float(is_ip(host)),             # host가 IP 주소인지 (0/1)

        # 길이 특성 (피싱 URL은 보통 길고 복잡함)
        "host_len":              float(len(host)),               # 호스트 문자열 길이
        "url_len":               float(len(raw)),                # 전체 URL 길이
        "path_len":              float(len(path)),               # 경로(path) 길이
        "query_len":             float(len(query)),              # 쿼리스트링 길이

        # 특수문자 개수 (피싱 URL은 특수문자가 많은 경향)
        "dot_count":             float(raw.count(".")),          # 점(.) 개수
        "hyphen_count":          float(raw.count("-")),          # 하이픈(-) 개수
        "hyphen_in_host":        float(1 if "-" in host else 0), # 호스트에 하이픈 포함 여부
        "slash_count":           float(raw.count("/")),          # 슬래시(/) 개수

        # 숫자 관련 (피싱 URL은 랜덤 숫자를 포함하는 경우 많음)
        "digit_count":           float(sum(c.isdigit() for c in raw)),  # 숫자 문자 총 개수
        "digit_ratio":           round(float(digit_ratio), 4),           # 전체 URL 중 숫자 비율

        # 특수 기호
        "at_count":              float(raw.count("@")),          # @ 개수 (URL 우회 수법에 사용됨)

        # 도메인 구조
        "subdomain_count":       float(subdomain_count(host)),   # 서브도메인 깊이
        "host_main_len":         float(len(host_main)),          # 도메인 메인 파트 길이
        "vowel_count":           float(sum(c in "aeiou" for c in host_main)),  # 모음 개수 (적으면 랜덤 생성 도메인 느낌)

        # 위험 패턴
        "has_shortener":         float(1 if host in SHORTENER_DOMAINS else 0),  # 단축 URL 서비스 도메인 여부
        "risky_tld":             float(1 if tld in RISKY_TLDS else 0),          # 위험 TLD 여부

        # 키워드 기반
        "suspicious_word_count": float(suspicious_word_count(raw)),  # 피싱 의심 키워드 포함 수

        # 인코딩
        "encoded_ratio":         round(float(encoded_ratio), 4),     # URL 인코딩(%) 문자 비율

        # 단축 URL 패턴 유사도
        "shortener_style_score": float(shortener_style_score(host, path)),  # 0~4점
    }


# ─────────────────────────────────────────────────────────
# 모델 번들 클래스
# ─────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    """
    학습된 모델 4개를 하나로 묶어 관리하는 데이터 클래스

    @dataclass: __init__, __repr__ 등을 자동 생성해주는 데코레이터
    4개를 묶어서 저장/로드해야 예측 시 학습 때와 동일한 변환 방식을 유지할 수 있음
    (예: 학습 때 fit한 TF-IDF 사전과 StandardScaler를 예측 때도 동일하게 사용해야 함)

    model    : 학습된 LogisticRegression 모델
    host_vec : host 문자열용 TF-IDF 벡터라이저 (학습 시 fit된 상태로 저장)
    path_vec : path+query 문자열용 TF-IDF 벡터라이저
    scaler   : 수치 피처 정규화용 StandardScaler (학습 시 fit된 상태로 저장)
    """
    model:    Any
    host_vec: Any
    path_vec: Any
    scaler:   Any

    def to_dict(self) -> dict[str, Any]:
        """pkl 파일로 저장하기 위해 딕셔너리로 변환"""
        return {
            "model":    self.model,
            "host_vec": self.host_vec,
            "path_vec": self.path_vec,
            "scaler":   self.scaler,
        }


# ─────────────────────────────────────────────────────────
# ML 패키지 확인
# ─────────────────────────────────────────────────────────
def require_ml_packages() -> None:
    """
    ML 관련 패키지가 설치되어 있는지 확인
    없으면 어떤 패키지가 없는지 명확한 에러 메시지 출력
    """
    try:
        import joblib  # noqa: F401  모델 저장/로드
        import pandas  # noqa: F401  데이터프레임
        import scipy   # noqa: F401  희소 행렬 (sparse matrix)
        import sklearn # noqa: F401  ML 알고리즘
    except ImportError as exc:
        raise ImportError(
            "ML functions require joblib, pandas, scipy, and scikit-learn."
        ) from exc


# ─────────────────────────────────────────────────────────
# 모델 저장/로드 함수
# ─────────────────────────────────────────────────────────
def load_model_bundle(model_path: Path | str = DEFAULT_MODEL_PATH) -> ModelBundle:
    """
    pkl 파일에서 모델 번들 로드 후 ModelBundle 객체로 반환

    joblib을 사용하는 이유: pickle보다 numpy 배열(TF-IDF 행렬 등) 직렬화가 빠르고 효율적
    로드 후 필수 키(model, host_vec, path_vec, scaler)가 모두 있는지 검증
    """
    require_ml_packages()
    import joblib

    bundle = joblib.load(Path(model_path))

    # 타입 검증: dict 형태여야 함
    if not isinstance(bundle, dict):
        raise TypeError("Saved model file must contain a dictionary bundle.")

    # 필수 키 검증 (set 차집합으로 누락된 키 찾기)
    missing = {"model", "host_vec", "path_vec", "scaler"} - set(bundle)
    if missing:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing)}")

    return ModelBundle(
        model=bundle["model"],
        host_vec=bundle["host_vec"],
        path_vec=bundle["path_vec"],
        scaler=bundle["scaler"],
    )


def save_model_bundle(
    bundle: ModelBundle,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    """
    모델 번들을 pkl 파일로 저장

    ModelBundle → to_dict() → pkl 파일
    저장 폴더가 없으면 자동 생성 (mkdir parents=True)
    """
    require_ml_packages()
    import joblib

    chosen_path = Path(model_path)
    chosen_path.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성
    joblib.dump(bundle.to_dict(), chosen_path)
    return chosen_path


# 외부에서 from url_phishing_common import * 할 때 노출할 항목 목록
__all__ = [
    "DEFAULT_DB_PATH", "DEFAULT_MODEL_PATH", "DEFAULT_PHISH_DATASET_PATH",
    "DEFAULT_THRESHOLD", "DEFAULT_TRANCO_PATH", "LEGIT_BRANDS", "LEGIT_PATHS",
    "MODULE_DIR", "ModelBundle", "NUMERIC_FEATURE_COLUMNS", "RANDOM_STATE",
    "RISKY_TLDS", "SHORTENER_DOMAINS", "SUSPICIOUS_WORDS", "TYPOSQUATTING_THRESHOLD",
    "extract_features", "get_tld", "is_ip", "is_typosquatting", "load_model_bundle",
    "normalize_domain", "normalize_url", "parse_parts", "require_ml_packages",
    "save_model_bundle", "shortener_style_score", "subdomain_count",
    "suspicious_word_count",
]
