"""
url_phishing_train.py
──────────────────────
URL 피싱 탐지 모델 학습 모듈

주요 역할:
1. PhiUSIIL 피싱 데이터 + Tranco 정상 URL 데이터 로드
2. 1:1 균형 샘플링
3. TF-IDF char n-gram + 수치 피처로 학습 데이터 구성
4. LogisticRegression 학습 및 평가
5. 모델 번들(model, host_vec, path_vec, scaler) 저장
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

# url_phishing_common.py에서 공통 상수/함수 가져오기
from url_phishing_common import (
    DEFAULT_MODEL_PATH,           # 모델 저장 경로 (url_model_v2.pkl)
    DEFAULT_PHISH_DATASET_PATH,   # 피싱 데이터 경로
    DEFAULT_TRANCO_PATH,          # 정상 도메인 데이터 경로
    LEGIT_PATHS,                  # 정상 URL 생성용 경로 패턴 15개
    ModelBundle,                  # 모델 4개를 묶는 데이터 클래스
    NUMERIC_FEATURE_COLUMNS,      # 수치 피처 컬럼 순서 (20개)
    RANDOM_STATE,                 # 랜덤 시드 (42)
    SHORTENER_DOMAINS,            # 단축 URL 도메인 목록
    extract_features,             # URL → 수치 피처 20개 추출 함수
    normalize_url,                # URL 정규화 함수
    parse_parts,                  # URL 파싱 함수
    require_ml_packages,          # ML 패키지 설치 확인
    save_model_bundle,            # 모델 번들 저장 함수
)


# ─────────────────────────────────────────────────────────
# 정상 URL 생성 함수
# ─────────────────────────────────────────────────────────
def generate_normal_urls(
    domain_list: Sequence[str],
    paths: Sequence[str] = LEGIT_PATHS,
    per_domain: int = 4,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """
    Tranco 도메인 목록에서 정상 URL을 생성

    피싱 데이터와 균형을 맞추기 위한 정상 URL 생성 전략:
    - 각 도메인당 https://domain, https://www.domain 기본 2개 생성
    - 추가로 LEGIT_PATHS 중 per_domain개를 랜덤 선택해서 조합
    - 단축 URL 도메인은 제외 (학습 데이터 오염 방지)

    예) domain = "naver.com", per_domain = 4 일 때:
        "https://naver.com"
        "https://www.naver.com"
        "https://naver.com/about"
        "https://naver.com/help"
        "https://naver.com/products"
        "https://naver.com/news"

    Parameters
    ----------
    domain_list  : Tranco 도메인 목록
    paths        : 조합할 경로 패턴 목록
    per_domain   : 도메인당 추가 생성할 URL 수
    random_state : 재현성을 위한 랜덤 시드
    """
    rng = random.Random(random_state)  # 재현 가능한 랜덤 객체 생성
    urls: list[str] = []

    for domain in domain_list:
        domain = str(domain).strip().lower()

        # 빈 도메인이나 단축 URL 도메인은 건너뜀
        if not domain or domain in SHORTENER_DOMAINS:
            continue

        # 기본 2개: http, https
        urls.append(f"https://{domain}")
        urls.append(f"https://www.{domain}")

        # 랜덤으로 per_domain개의 경로를 선택해서 조합
        sample_size = min(per_domain, len(paths))
        for path in rng.sample(list(paths), k=sample_size):
            urls.append(f"https://{domain}{path}")

    return urls


# ─────────────────────────────────────────────────────────
# 메인 학습 함수
# ─────────────────────────────────────────────────────────
def train_url_model(
    phish_path: Path | str = DEFAULT_PHISH_DATASET_PATH,
    tranco_path: Path | str = DEFAULT_TRANCO_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    normal_domain_limit: int = 50_000,
    per_domain: int = 4,
    save_model: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    피싱 URL 탐지 모델 학습 전체 파이프라인

    학습 순서:
    1. 피싱 데이터 로드 (PhiUSIIL)
    2. 정상 URL 생성 (Tranco 도메인 기반)
    3. 1:1 균형 샘플링 + 셔플
    4. TF-IDF 벡터화 (host, path 각각)
    5. 수치 피처 추출 + StandardScaler 정규화
    6. 피처 결합 (hstack): TF-IDF host + TF-IDF path + 수치 피처
    7. LogisticRegression 학습
    8. 평가 및 모델 저장

    Parameters
    ----------
    phish_path          : 피싱 데이터 CSV 경로
    tranco_path         : Tranco 도메인 CSV 경로
    model_path          : 모델 저장 경로
    normal_domain_limit : 정상 도메인 최대 사용 수
    per_domain          : 도메인당 생성할 정상 URL 수
    save_model          : 모델 파일 저장 여부
    random_state        : 랜덤 시드

    Returns
    -------
    metrics 딕셔너리 (정확도, F1, 분류 리포트, 혼동 행렬, 모델 번들 포함)
    """
    require_ml_packages()  # ML 패키지 설치 여부 확인

    # ML 관련 라이브러리는 함수 내부에서 import (최초 실행 시에만 로드)
    import pandas as pd
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # ── 1. 피싱 데이터 로드 ──────────────────────────────
    phish_df = pd.read_csv(Path(phish_path))[["URL", "label"]].copy()
    phish_df["URL"] = phish_df["URL"].astype(str).fillna("").str.strip()
    phish_df = phish_df[phish_df["URL"] != ""].reset_index(drop=True)

    # label == 0 인 행만 피싱으로 사용 (데이터셋에서 0이 피싱을 의미)
    phish_df = phish_df[phish_df["label"] == 0].copy()
    phish_df["label"] = 0  # 피싱 라벨을 0으로 통일

    # URL 정규화 + 중복 제거
    phish_df["URL"] = phish_df["URL"].apply(normalize_url)
    phish_df = phish_df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    # ── 2. 정상 URL 생성 (Tranco 도메인 기반) ────────────
    tranco_df = pd.read_csv(Path(tranco_path))

    # Tranco CSV는 [순위, 도메인] 구조이므로 두 번째 컬럼이 도메인
    domain_col = tranco_df.columns[1] if tranco_df.shape[1] >= 2 else tranco_df.columns[0]
    domains = (
        tranco_df[domain_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .pipe(lambda series: series[series != ""])  # 빈 문자열 제거
        .drop_duplicates()
        .tolist()
    )[:normal_domain_limit]  # 상위 50,000개만 사용

    # Tranco 도메인 + LEGIT_PATHS 조합으로 정상 URL 생성
    normal_urls = generate_normal_urls(
        domain_list=domains,
        paths=LEGIT_PATHS,
        per_domain=per_domain,
        random_state=random_state,
    )
    normal_df = pd.DataFrame({"URL": normal_urls, "label": 1})  # 정상 라벨 = 1
    normal_df["URL"] = normal_df["URL"].apply(normalize_url)
    normal_df = normal_df.drop_duplicates(subset=["URL"]).reset_index(drop=True)

    # ── 3. 1:1 균형 샘플링 ───────────────────────────────
    # 더 적은 쪽 수에 맞춰 언더샘플링 (클래스 불균형 해소)
    sample_size = min(len(phish_df), len(normal_df))
    if sample_size == 0:
        raise ValueError("Training data is empty after preprocessing.")

    phish_df  = phish_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    normal_df = normal_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # 피싱 + 정상 합치고 셔플 (순서 편향 방지)
    dataset = pd.concat([phish_df, normal_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # URL을 파싱해서 host, path, query 컬럼 추가 (TF-IDF 입력용)
    dataset[["raw_url", "host", "path", "query"]] = dataset["URL"].apply(
        lambda url: pd.Series(parse_parts(url))
    )

    # ── 4. Train/Test 분리 (8:2) ─────────────────────────
    # stratify=label: 피싱/정상 비율을 train/test에서 동일하게 유지
    x_train_df, x_test_df, y_train, y_test = train_test_split(
        dataset,
        dataset["label"],
        test_size=0.2,
        random_state=random_state,
        stratify=dataset["label"],
    )

    # ── 5. TF-IDF 벡터화 (텍스트 피처) ──────────────────
    # char n-gram: URL 문자열을 3~5글자 단위로 쪼개서 패턴 학습
    # sublinear_tf=True: 고빈도 패턴의 편향을 로그 스케일로 완화
    host_vec = TfidfVectorizer(
        analyzer="char",        # 문자(character) 단위 분석
        ngram_range=(3, 5),     # 3글자 ~ 5글자 조합 사용
        min_df=2,               # 2번 이상 등장한 패턴만 피처로 사용 (희귀 패턴 제거)
        max_features=30_000,    # 최대 30,000개 패턴 (호스트용)
        sublinear_tf=True,      # TF에 log 적용 (고빈도 편향 완화)
    )
    path_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20_000,    # 최대 20,000개 패턴 (경로용)
        sublinear_tf=True,
    )

    # fit_transform: 학습 데이터로 패턴 사전 구축 + 변환
    x_train_host = host_vec.fit_transform(x_train_df["host"])
    # transform만: 테스트 데이터는 학습 때 만든 사전으로 변환만 (fit 안 함)
    x_test_host  = host_vec.transform(x_test_df["host"])

    # path + query를 합쳐서 경로 텍스트 구성
    train_path_text = x_train_df["path"].fillna("") + "?" + x_train_df["query"].fillna("")
    test_path_text  = x_test_df["path"].fillna("") + "?" + x_test_df["query"].fillna("")
    x_train_path = path_vec.fit_transform(train_path_text)
    x_test_path  = path_vec.transform(test_path_text)

    # ── 6. 수치 피처 추출 + 정규화 ───────────────────────
    # extract_features()로 URL당 수치 피처 20개 추출
    train_num = pd.DataFrame(
        [extract_features(url) for url in x_train_df["URL"]],
        columns=NUMERIC_FEATURE_COLUMNS,  # 컬럼 순서 고정 (학습/예측 시 동일해야 함)
    )
    test_num = pd.DataFrame(
        [extract_features(url) for url in x_test_df["URL"]],
        columns=NUMERIC_FEATURE_COLUMNS,
    )

    # StandardScaler: 평균 0, 표준편차 1로 정규화
    # fit_transform: 학습 데이터의 평균/표준편차를 계산하고 변환
    scaler = StandardScaler()
    x_train_num = csr_matrix(scaler.fit_transform(train_num))  # 희소 행렬로 변환 (메모리 효율)
    x_test_num  = csr_matrix(scaler.transform(test_num))       # 같은 평균/표준편차로 변환만

    # ── 7. 피처 결합 ─────────────────────────────────────
    # hstack: 수평으로 붙이기 (각 피처 행렬을 옆으로 이어 붙임)
    # 최종 차원: TF-IDF host(30,000) + TF-IDF path(20,000) + 수치(20) = 약 50,020차원
    x_train = hstack([x_train_host, x_train_path, x_train_num]).tocsr()
    x_test  = hstack([x_test_host,  x_test_path,  x_test_num]).tocsr()

    # ── 8. 모델 학습 ─────────────────────────────────────
    model = LogisticRegression(
        max_iter=2000,           # 수렴 보장을 위한 최대 반복 횟수
        C=1.0,                   # 역정규화 강도 (클수록 과적합 가능성 높음)
        class_weight="balanced", # 피싱/정상 불균형을 자동으로 보완
        solver="liblinear",      # 고차원 희소 행렬에 최적화된 솔버
        random_state=random_state,
    )
    model.fit(x_train, y_train)  # 학습 실행

    # ── 9. 평가 ──────────────────────────────────────────
    y_pred = model.predict(x_test)

    # 모델 번들: 학습된 4개 객체를 하나로 묶음 (저장/로드 편의성)
    bundle = ModelBundle(
        model=model,
        host_vec=host_vec,
        path_vec=path_vec,
        scaler=scaler,
    )

    # 평가 지표 딕셔너리
    metrics = {
        "train_size":            int(len(x_train_df)),
        "test_size":             int(len(x_test_df)),
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "f1_macro":              float(f1_score(y_test, y_pred, average="macro")),
        "f1_phishing":           float(f1_score(y_test, y_pred, pos_label=0)),  # 피싱 클래스(0) F1
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "bundle":                bundle,  # 모델 번들 포함
    }

    # ── 10. 모델 저장 ─────────────────────────────────────
    if save_model:
        saved_path = save_model_bundle(bundle, model_path)
        metrics["saved_model_path"] = str(saved_path)

    return metrics


# 외부 import 시 노출할 항목
__all__ = [
    "generate_normal_urls",
    "train_url_model",
]
