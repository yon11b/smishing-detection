※참고: SMS 스미싱 모델(smishing_kobert.pt)은 용량이 너무 커서 업로드가 안 됩니다!<br>
구글 드라이브에 전체 코드를 올려놓았으니 참고 바랍니다. [[링크]](https://drive.google.com/file/d/1mIfYnPC4qemr0fFbXxB0vC0x1b8aePXg/view?usp=drive_link)

## 실행 명령어

- `pip install -r requirements.txt`
- `streamlit run app.py`

## 개요

SMS 스미싱 탐지 모델입니다.
<img width="1333" height="836" alt="image" src="https://github.com/user-attachments/assets/98d8754f-d453-497e-b6ad-4f90afa0e430" />

## 구조도

<img width="1440" height="1736" alt="image" src="https://github.com/user-attachments/assets/7f8bdf26-cafc-4772-b99b-78a06a4ff470" />

## 협업 툴

Notion [[링크]](https://www.notion.so/1-32eb3aca18a48057b856ce8a53d4711a?source=copy_link)
<img width="804" height="476" alt="image" src="https://github.com/user-attachments/assets/4570af55-a587-4c93-8bd8-a66ab67551f4" />

## 기술스택

- SMS 스미싱 탐지 모델: KoBert
- URL 피싱 탐지 모델: TF-IDF/Logistic Regression
- LLM: GPT-4o (via OpenAI API)
- DB: sqlite
- front: streamlit
- websearch

## 팀 정보

<br>
<table align="center" width="700" border="1" cellspacing="0" cellpadding="10" style="border-collapse: collapse; text-align: center;">
  <thead style="background-color: #f2f2f2;">
    <tr>
      <th width="150">성명</th>
      <th width="550">역할</th>
    </tr> 
  </thead>
  <tbody>
    <tr>
      <td>김건하</td>
      <td>
        - 프로젝트 총괄<br>
        - URL 탐지 모델 학습
      </td>
    </tr>
    <tr>
      <td>김정현</td>
      <td>        
        - 데이터 전처리<br>
        - SMS 스미싱 탐지 유형분류 개발
      </td>
    </tr>
    <tr>
      <td>백하연</td>
      <td>
        - SMS 스미싱 탐지 모델 학습(파인튜닝)<br>
        - GPT 에이전트 개발
      </td>
    </tr>
    <tr>
      <td>우수아</td>
      <td>
        - Web Search 기능 구현<br>
        - 코드 통합 관리
      </td>
    </tr>
    <tr>
      <td>유인기</td>
      <td>
        - 모델 테스팅<br>
        - 대응 가이드 조사
      </td>
    </tr>
    <tr>
      <td>조수호</td>
      <td>
        - URL 탐지 모델 개발<br>
        - UI-모델 통합 코드 작성 / 기능 추가
      </td>
    </tr>
    <tr>
      <td>최원기</td>
      <td>
        - UI 화면 구성<br>
        - 최종 발표
      </td>
    </tr>
  </tbody>
</table>
<br>
