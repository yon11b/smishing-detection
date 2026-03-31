"""
url_phishing_predict.py
────────────────────────
URL 피싱 탐지기 모듈

주요 역할:
1. SQLite 블랙리스트 DB 관리 (KISA 피싱 도메인 저장/조회)
2. URLPhishingDetector 클래스: 다층 탐지 파이프라인 실행
   - 1단계: KISA 블랙리스트 조회
   - 2단계: IP 주소 탐지
   - 3단계: 단축 URL 탐지
   - 4단계: 타이포스쿼팅 감지
   - 5단계: ML 모델 판단
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Iterable, Optional

# url_phishing_common.py에서 공통 함수/상수 가져오기
from url_phishing_common import (
    DEFAULT_DB_PATH,         # 블랙리스트 DB 경로
    DEFAULT_MODEL_PATH,      # 모델 pkl 경로
    DEFAULT_THRESHOLD,       # 판정 임계값 (0.60)
    ModelBundle,             # 모델 번들 데이터 클래스
    NUMERIC_FEATURE_COLUMNS, # 수치 피처 컬럼 순서
    SHORTENER_DOMAINS,       # 단축 URL 도메인 목록
    extract_features,        # URL → 수치 피처 추출
    is_ip,                   # IP 주소 여부 확인
    is_typosquatting,        # 타이포스쿼팅 감지
    load_model_bundle,       # pkl 파일에서 모델 로드
    normalize_domain,        # URL에서 도메인만 추출
    parse_parts,             # URL 파싱 (raw, host, path, query)
    require_ml_packages,     # ML 패키지 설치 확인
)


# ─────────────────────────────────────────────────────────
# SQLite 블랙리스트 DB 관리 함수
# ─────────────────────────────────────────────────────────

def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    """
    SQLite 블랙리스트 DB 초기화
    테이블이 없으면 생성, 이미 있으면 그대로 사용 (CREATE TABLE IF NOT EXISTS)

    테이블 구조:
        id       : 자동 증가 정수 (기본 키)
        domain   : 피싱 도메인 (중복 불가, UNIQUE)
        source   : 출처 (기본값: 'KISA')
        added_at : 추가 시각 (자동 기록)
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)  # DB 파일 폴더 없으면 생성

    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                domain   TEXT NOT NULL UNIQUE,
                source   TEXT DEFAULT 'KISA',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.commit()
    return db_path


def add_domain(
    domain: str,
    source: str = "KISA",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Optional[str]:
    """
    블랙리스트에 단일 도메인 추가
    이미 존재하면 무시 (INSERT OR IGNORE)

    Returns: 정규화된 도메인 문자열, 빈 도메인이면 None
    """
    normalized = normalize_domain(domain)  # URL에서 host 부분만 추출
    if not normalized:
        return None

    with sqlite3.connect(init_db(db_path)) as con:
        con.execute(
            "INSERT OR IGNORE INTO blacklist (domain, source) VALUES (?, ?)",
            (normalized, source),
        )
        con.commit()
    return normalized


def add_domains_bulk(
    domains: Iterable[str],
    source: str = "KISA",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """
    블랙리스트에 여러 도메인 일괄 추가 (executemany로 한 번에 처리)

    KISA CSV 파일에서 읽어온 도메인 목록을 한 번에 넣을 때 사용
    Returns: 추가 시도한 도메인 수
    """
    # 도메인 정규화 후 빈 문자열 제거
    rows = [(normalize_domain(d), source) for d in domains if normalize_domain(d)]
    if not rows:
        return 0

    with sqlite3.connect(init_db(db_path)) as con:
        # executemany: 여러 행을 한 번에 삽입 (반복 INSERT보다 훨씬 빠름)
        con.executemany(
            "INSERT OR IGNORE INTO blacklist (domain, source) VALUES (?, ?)", rows
        )
        con.commit()
    return len(rows)


def is_blacklisted(
    url_or_domain: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """
    URL 또는 도메인이 블랙리스트에 있는지 확인

    서브도메인 포함 체크:
        "evil.paypal-secure.ml" 입력 시
        1) "paypal-secure.ml" 정확 매칭 체크
        2) "secure.ml" 부모 도메인 체크
        → 하나라도 일치하면 True 반환

    LIMIT 1: 하나만 찾으면 되므로 전체 탐색 불필요 (성능 최적화)
    """
    host = normalize_domain(url_or_domain)
    if not host:
        return False

    with sqlite3.connect(init_db(db_path)) as con:
        # 1단계: 정확 일치 체크
        if con.execute(
            "SELECT 1 FROM blacklist WHERE domain = ? LIMIT 1", (host,)
        ).fetchone():
            return True

        # 2단계: 서브도메인 체크
        # "a.b.evil.com" → ["a", "b", "evil", "com"]
        # range(1, len-1): 인덱스 1부터 마지막 전까지 (TLD 단독은 제외)
        parts = host.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])  # 상위 도메인 조합
            if con.execute(
                "SELECT 1 FROM blacklist WHERE domain = ? LIMIT 1", (parent,)
            ).fetchone():
                return True

    return False


def list_all(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """전체 블랙리스트 조회 (최근 추가 순)"""
    with sqlite3.connect(init_db(db_path)) as con:
        rows = con.execute(
            "SELECT id, domain, source, added_at FROM blacklist ORDER BY added_at DESC"
        ).fetchall()
    return [{"id": r[0], "domain": r[1], "source": r[2], "added_at": r[3]} for r in rows]


def remove_domain(
    domain: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Optional[str]:
    """블랙리스트에서 도메인 삭제"""
    normalized = normalize_domain(domain)
    if not normalized:
        return None

    with sqlite3.connect(init_db(db_path)) as con:
        con.execute("DELETE FROM blacklist WHERE domain = ?", (normalized,))
        con.commit()
    return normalized


# ─────────────────────────────────────────────────────────
# URL 피싱 탐지기 클래스
# ─────────────────────────────────────────────────────────

class URLPhishingDetector:
    """
    다층 탐지 파이프라인을 실행하는 URL 피싱 탐지기

    탐지 순서 (classify_stage):
        1. KISA 블랙리스트 → 피싱 (확률 1.0)
        2. IP 주소 host   → 피싱 (확률 1.0)
        3. 단축 URL       → 의심 (확률 0.6)
        4. 타이포스쿼팅   → 피싱 (확률 0.95)
        5. ML 모델        → 확률 기반 분류

    ML 모델은 명백한 케이스(블랙리스트, IP 등)를 처리하고 남은
    99.3%의 URL을 직접 판단하는 ML 중심 구조
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        auto_load: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        db_path    : 블랙리스트 SQLite DB 경로
        model_path : 학습된 모델 pkl 경로
        threshold  : ML 판정 임계값 (기본 0.60)
        auto_load  : 초기화 시 모델 자동 로드 여부
        """
        self.db_path    = init_db(db_path)      # DB 초기화 (없으면 생성)
        self.model_path = Path(model_path)
        self.threshold  = threshold
        self.bundle: Optional[ModelBundle] = None           # 로드된 모델 번들
        self.model_load_error: Optional[Exception] = None  # 모델 로드 실패 시 에러 저장

        # auto_load=True이고 pkl 파일이 존재하면 자동으로 모델 로드
        if auto_load and self.model_path.exists():
            try:
                self.load_model()
            except Exception as exc:
                self.model_load_error = exc  # 로드 실패해도 앱이 죽지 않도록 에러 저장

    def load_model(self, model_path: Path | str | None = None) -> ModelBundle:
        """
        pkl 파일에서 모델 번들 로드

        model_path가 None이면 초기화 시 지정한 경로 사용
        """
        chosen_path = Path(model_path) if model_path is not None else self.model_path
        self.bundle = load_model_bundle(chosen_path)
        self.model_path = chosen_path
        self.model_load_error = None
        return self.bundle

    def classify_stage(self, url: str) -> str:
        """
        URL이 어느 단계에서 처리될지 결정

        반환값: "invalid" | "blacklist" | "ip" | "shortener" | "typosquatting" | "ml"

        순서가 중요: 명백한 케이스부터 걸러내고 나머지만 ML로 보냄
        실제로 ML에 도달하는 비율: 약 99.3%
        """
        raw, host, _, _ = parse_parts(url)

        # 유효하지 않은 URL
        if not raw or not host:
            return "invalid"

        # 1단계: KISA 블랙리스트 (서브도메인 포함 확인)
        if is_blacklisted(host, self.db_path):
            return "blacklist"

        # 2단계: IP 주소 host (도메인 없이 직접 IP로 접속하는 피싱)
        if is_ip(host):
            return "ip"

        # 3단계: 단축 URL (bit.ly, tinyurl.com 등)
        if host in SHORTENER_DOMAINS:
            return "shortener"

        # 4단계: 타이포스쿼팅 (유명 브랜드와 유사한 도메인)
        if is_typosquatting(host):
            return "typosquatting"

        # 5단계: ML 모델로 판단
        return "ml"

    def predict_url(self, url: str) -> dict[str, Any]:
        """
        URL 피싱 여부 예측

        Returns
        -------
        {
            "url":          str,            입력 URL
            "host":         str,            파싱된 host
            "prediction":   str,            "피싱" | "의심" | "정상" | "오류"
            "prob_phishing": float | None,  피싱 확률 (0~1)
            "prob_legit":    float | None,  정상 확률 (0~1)
            "reason":       str,            판단 근거
            "stage":        str,            처리 단계
        }
        """
        raw, host, path, query = parse_parts(url)

        # 유효하지 않은 URL 처리
        if not raw or not host:
            return {
                "url": url, "host": host,
                "prediction": "오류",
                "prob_phishing": None, "prob_legit": None,
                "reason": "유효한 URL이 아닙니다.",
                "stage": "invalid",
            }

        # 단계 분류
        stage = self.classify_stage(raw)

        # ── 1단계: KISA 블랙리스트 ──────────────────────
        if stage == "blacklist":
            return {
                "url": url, "host": host,
                "prediction": "피싱",
                "prob_phishing": 1.0, "prob_legit": 0.0,
                "reason": "KISA 블랙리스트",
                "stage": stage,
            }

        # ── 2단계: IP 주소 ───────────────────────────────
        if stage == "ip":
            return {
                "url": url, "host": host,
                "prediction": "피싱",
                "prob_phishing": 1.0, "prob_legit": 0.0,
                "reason": "IP 주소 host",
                "stage": stage,
            }

        # ── 3단계: 단축 URL ──────────────────────────────
        # None 대신 의심 수준 고정 확률 반환 (SMS 모델 연동 시 prob_phishing이 필요하기 때문)
        if stage == "shortener":
            return {
                "url": url, "host": host,
                "prediction": "의심",
                "prob_phishing": 0.6, "prob_legit": 0.4,
                "reason": "단축 URL 도메인",
                "stage": stage,
            }

        # ── 4단계: 타이포스쿼팅 ─────────────────────────
        if stage == "typosquatting":
            return {
                "url": url, "host": host,
                "prediction": "피싱",
                "prob_phishing": 0.95, "prob_legit": 0.05,
                "reason": "유명 도메인 사칭 의심 (타이포스쿼팅)",
                "stage": stage,
            }

        # ── 5단계: ML 모델 판단 ──────────────────────────
        require_ml_packages()
        bundle = self._require_bundle()

        import pandas as pd
        from scipy.sparse import csr_matrix, hstack

        # TF-IDF 벡터화 (학습 때 fit된 벡터라이저로 transform만 수행)
        host_x = bundle.host_vec.transform([host])
        path_x = bundle.path_vec.transform([f"{path}?{query}"])

        # 수치 피처 추출 + 정규화
        num_df = pd.DataFrame(
            [extract_features(raw)], columns=NUMERIC_FEATURE_COLUMNS
        )
        num_x = csr_matrix(bundle.scaler.transform(num_df))

        # 피처 결합: TF-IDF host + TF-IDF path + 수치 피처
        x = hstack([host_x, path_x, num_x]).tocsr()

        # 확률 예측
        probs = bundle.model.predict_proba(x)[0]
        classes = list(bundle.model.classes_)
        prob_phishing = round(float(probs[classes.index(0)]), 4)  # 클래스 0 = 피싱
        prob_legit    = round(float(probs[classes.index(1)]), 4)  # 클래스 1 = 정상

        # 임계값 기반 최종 판정
        if prob_legit >= self.threshold:
            prediction = "정상"
        elif prob_phishing >= self.threshold:
            prediction = "피싱"
        else:
            prediction = "의심"  # 어느 쪽도 임계값 미달이면 의심

        return {
            "url": url, "host": host,
            "prediction": prediction,
            "prob_phishing": prob_phishing,
            "prob_legit": prob_legit,
            "reason": "ML 모델",
            "stage": stage,
        }

    def predict_urls(self, urls: Iterable[str]) -> list[dict[str, Any]]:
        """여러 URL을 한 번에 예측"""
        return [self.predict_url(url) for url in urls]

    # ─────────────────────────────────────────────────────
    # SMS 모델 연동용 메서드
    # ─────────────────────────────────────────────────────

    def get_url_prob(self, url: Optional[str]) -> float:
        """
        URL 피싱 확률(0~1)만 반환
        phishing_pipeline.py에서 SMS 확률과 합산할 때 사용

        prob_phishing이 None인 경우 (단축 URL 등)
        prediction 문자열로 대체 확률 반환
        """
        if not url:
            return 0.0

        result = self.predict_url(url)
        prob = result.get("prob_phishing")
        if prob is not None:
            return prob

        # prob_phishing이 None이면 prediction으로 대체
        return {"피싱": 1.0, "의심": 0.6, "정상": 0.0, "오류": 0.5}.get(
            result["prediction"], 0.5
        )

    def get_final_result(
        self,
        sms_prob: float,
        url: Optional[str] = None,
        url_weight: float = 0.7,
        sms_weight: float = 0.3,
    ) -> dict[str, Any]:
        """
        SMS 모델 확률 + URL 모델 확률 → 가중 평균으로 최종 위험도 산출

        Parameters
        ----------
        sms_prob   : SMS 모델이 뱉는 피싱 확률 (0~1)
        url        : SMS에서 추출한 URL (없으면 None)
        url_weight : URL 가중치 (기본 0.7) → URL이 더 중요
        sms_weight : SMS 가중치 (기본 0.3)
        """
        url_result = self.predict_url(url) if url else None
        url_prob   = self.get_url_prob(url)

        # 가중 평균으로 최종 위험도 계산
        risk_score = url_prob * url_weight + sms_prob * sms_weight

        if   risk_score >= 0.6:  verdict = "악성"
        elif risk_score >= 0.35: verdict = "의심"
        else:                    verdict = "정상"

        return {
            "verdict":    verdict,
            "risk_score": round(risk_score * 100, 1),  # 퍼센트로 변환
            "pct_danger": round(risk_score * 100, 1),
            "pct_normal": round((1 - risk_score) * 100, 1),
            "url_prob":   round(url_prob, 4),
            "sms_prob":   round(sms_prob, 4),
            "url_result": url_result,
        }

    def _require_bundle(self) -> ModelBundle:
        """
        모델 번들이 로드되어 있는지 확인하고 반환
        없으면 다시 로드 시도, 그래도 없으면 RuntimeError 발생
        """
        if self.bundle is not None:
            return self.bundle

        # 번들이 없으면 다시 로드 시도
        if self.model_path.exists():
            try:
                return self.load_model()
            except Exception as exc:
                self.model_load_error = exc

        # 로드 실패 시 명확한 에러 메시지
        msg = f"ML model bundle not available: {self.model_path}"
        if self.model_load_error:
            msg += f" | {self.model_load_error}"
        raise RuntimeError(msg)


# 외부 import 시 노출할 항목
__all__ = [
    "URLPhishingDetector",
    "add_domain", "add_domains_bulk",
    "init_db", "is_blacklisted",
    "list_all", "remove_domain",
]
