import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kobert_transformers import get_tokenizer, get_kobert_model
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = "smishing_kobert.pt"
THRESHOLD  = 0.5

smishing_keywords = [
    # 금융/계좌
    '계좌', '통장', '입금', '출금', '송금', '이체', '결제', '승인',
    '카드', '대출', '환급', '납부', '미납', '연체', '청구', '요금',
    '수수료', '이자', '원금', '잔액', '한도', '증액', '해지',

    # 본인인증/보안
    '인증', '본인확인', '본인인증', '보안', '개인정보', '비밀번호',
    '아이디', '로그인', '해킹', '도용', '유출', '탈취', '피싱',
    '악성', '바이러스', '앱설치', '업데이트', '최신버전',

    # 계정/서비스 상태
    '정지', '차단', '해제', '만료', '제한', '중단', '일시정지',
    '비활성', '잠금', '탈퇴', '삭제', '소멸', '만료일', '기한',
    '기간', '마감', '오늘까지', '오늘중',

    # 기관 사칭
    '금감원', '금융감독원', '금융위', '경찰', '검찰', '법원',
    '국세청', '건강보험', '국민연금', '행정안전부', '정부24',
    '국민건강', '복지부', '교육부', '관세청', '우체국',

    # 택배/배송
    '택배', '배송', '반송', '주소불명', '수령', '미수령', '통관',
    '반품', '교환', '배달', '도착', '출발', '운송장',

    # 사칭/긴급
    '긴급', '즉시', '빨리', '신속', '급하게', '지금바로',
    '명의', '사칭', '도용', '불법', '피해', '신고', '고소',
    '소송', '압류', '구속', '체포', '수사', '당장',

    # 가족사칭
    '액정', '수리', '핸드폰', '휴대폰', '폰', '임시번호',
    '엄마', '아빠', '부모님', '친구', '급하게',

    # 당첨/이벤트
    '당첨', '축하', '선물', '혜택', '이벤트', '무료', '공짜',
    '지원금', '보조금', '장려금', '환급금', '미지급', '미수령',

    # 링크/앱
    '링크', '클릭', '접속', '사이트', '홈페이지', '앱',
    '설치', '다운로드', '확인바랍니다', '확인요망',
]


# ===============================
# 1. 전처리1
# ===============================
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\w\s가-힣.,!?%]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_real_smishing(text):
    return any(kw in text for kw in smishing_keywords)


# ===============================
# 2. 데이터셋
# ===============================
class SmishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "token_type_ids": inputs["token_type_ids"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ===============================
# 3. 모델
# ===============================
class SmishingClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.kobert = get_kobert_model()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.kobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = outputs[0][:, 0, :]
        return self.classifier(cls)


# ===============================
# 4. 학습 / 평가 함수
# ===============================
def _train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct = 0, 0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)


def _evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(dataloader), correct / len(dataloader.dataset), all_preds, all_labels


# ===============================
# 5. 학습 실행
# ===============================
def _train_model(device, tokenizer):
    print("[kobert] 저장된 모델 없음 → 학습 시작")

    df = pd.read_csv("output.csv")

    # 전처리2
    df = df.dropna(subset=["content", "class"])
    df = df[df["class"] != "class"]             # 헤더 중복 행 제거
    df["content"] = df["content"].astype(str)
    df["class"]   = df["class"].astype(float).astype(int)
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.strip() != ""].drop_duplicates(subset=["content"])

    # class=1 진짜 스미싱만 필터링(전처리3)
    df_class0 = df[df["class"] == 0]
    df_class1 = df[df["class"] == 1]
    df_class1_filtered = df_class1[df_class1["content"].apply(_is_real_smishing)]
    print(f"class=1 필터링 전: {len(df_class1)}개 → 후: {len(df_class1_filtered)}개")

    # 클래스 균형 맞추기(언더샘플링)
    n = min(len(df_class0), len(df_class1_filtered), 5000)
    df_class0_sampled = df_class0.sample(n=n, random_state=42)
    df_class1_sampled = df_class1_filtered.sample(n=n, random_state=42)
    df = pd.concat([df_class0_sampled, df_class1_sampled], ignore_index=True).sample(frac=1, random_state=42)

    print(f"전체 데이터: {len(df)}개\n{df['class'].value_counts()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["class"])

    overlap = pd.merge(train_df, test_df, on="content")
    print(f"겹치는 데이터: {len(overlap)}개")

    train_dataset = SmishingDataset(train_df["content"], train_df["class"], tokenizer)
    test_dataset  = SmishingDataset(test_df["content"],  test_df["class"],  tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

    EPOCHS = 3
    model     = SmishingClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, _, _ = _evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    _, _, preds, labels = _evaluate(model, test_loader, criterion, device)
    print("\n[분류 리포트]")
    print(classification_report(labels, preds, target_names=["정상(0)", "스미싱(1)"]))

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[kobert] 모델 저장 완료 → {MODEL_PATH}")

    return model


# ===============================
# 6. 초기화 (모델 로드 or 학습)
# ===============================
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = get_tokenizer()
_model     = SmishingClassifier().to(_device)

if os.path.exists(MODEL_PATH):
    print(f"[kobert] 저장된 모델 로드 → {MODEL_PATH}")
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
else:
    _model = _train_model(_device, _tokenizer)

_model.eval()


# ===============================
# 7. 추론 함수 (외부에서 import해서 사용)
# ===============================
def predict(text: str) -> dict:
    """
    SMS 텍스트를 받아 악성 여부와 신뢰도를 반환합니다.

    Returns
    -------
    {
        "is_malicious": bool,
        "confidence":   float  # 0.0 ~ 1.0
    }
    """
    inputs = _tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = inputs["input_ids"].to(_device)
    attention_mask = inputs["attention_mask"].to(_device)
    token_type_ids = inputs["token_type_ids"].to(_device)

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask, token_type_ids)
        prob    = torch.softmax(outputs, dim=1)

    smishing_prob = prob[0][1].item()
    print(smishing_prob)

    return {
        "is_malicious": smishing_prob >= THRESHOLD,
        "confidence":   round(smishing_prob, 4),
    }