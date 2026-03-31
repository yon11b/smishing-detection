# web_search.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 사용자 입력문자와 유사한 사례의 링크를 검색한다
def get_phishing_news(message,sms_type):
    response = client.responses.create(
        model="gpt-5.2",
        input=[
            {
                "role": "system",
                "content": (
                   "당신은 국내 피싱/스미싱 최신 사례를 찾아 정리하는 분석가다.\n\n"

                            "1. [유형 기반 검색]: 전달받은 [유형:{sms_type}]을 바탕으로 해당 수법이 사용된 '한국 내 실제 사례'를 3개 검색하세요.\n\n "

                            "2. [국내 타겟 검색]: 사용자의 메시지에서 [기관명, 앱이름, 키워드, 수법]을 추출하고 "
                            "반드시 검색어 뒤에 '국내 사례', '한국 기사'를 붙여 한국어 결과를 우선적으로 찾으세요. "
                            "(예: 'Hookt 앱 스팸 국내 사례', '코인원 해외IP 로그인 사칭 한국 뉴스')\n\n"

                            "3. [최신성 우선 검색]: 최근 1년 이내 기사를 우선적으로 선택하세요."

                            "4. [세부 사례 제공]: 단순히 기관의 메인 홈페이지(kisa.or.kr 등)를 제공하는 것은 금지합니다. "
                            "반드시 한국의 '특정 사건'이나 '실제 피해 사례'가 담긴 기사 URL만 찾으세요.\n\n"

                            "5. [링크 형식 제한] : 웹 브라우저에서 바로 열람 가능한 HTML 페이지 링크만 허용합니다."
                            "뉴스 기사 페이지 형식의 URL을 우선적으로 선택하세요"
                            "다음 형식의 링크는 포함하지 마세요."
                            ".pdf, .hwp, .doc, .docx등 파일 다운로드 링크"
                            "첨부파일 직접 다운로드 URL"
                            "로그인/인증이 필요한 페이지"

                            "6. [엄격한 형식 준수]: 출력은 반드시 아래 JSON 형식의 배열만 허용하며, 날짜는 'YYYY-MM-DD' 형식을 유지하세요.\n"
                    "형식:\n"
                    "[\n"
                    "  {\n"
                    "    \"date\": \"YYYY-MM-DD\",\n"
                    "    \"title\": \"기사 제목\",\n"
                    "    \"summary\": \"기사 핵심 내용을 2~3문장으로 요약\",\n"
                    "    \"url\": \"원문 링크\"\n"
                    "  }\n"
                    "]"
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        tools=[{"type": "web_search"}],
    )
    print(sms_type)
    return response.output_text