import os
import sys
from dotenv import load_dotenv
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

class GenQuestion:
    def __init__(self, model):
        self.llm = model
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Bạn là một chuyên gia ngôn ngữ Hàn Quốc. Nhiệm vụ cùa bạn là tạo tài liệu phục vụ kỳ thi EPS theo hướng dẫn dưới đây:
            Hướng dẫn tạo tài liệu dựa trên các chủ đề sau đây
                - Đơn xin việc / Hồ sơ xin việc (입사지원서/이력서)
                - Hợp đồng lao động (근로 계약서)
                - Giấy xác nhận tuyển dụng (채용 확인서)
                - Giấy chứng nhận hoàn thành đào tạo (교육 수료증)
                - Giấy tờ chứng minh kinh nghiệm làm việc (경력 증명서)
                - Thư mời làm việc / thư xác nhận từ công ty (회사 초청장/확인서)
                - Giấy tờ xác nhận thu nhập (소득 증명서)
                - Tài liệu đăng ký tạm trú hoặc địa chỉ cư trú (거주지 등록 서류)
                - Tin nhắn xin nghỉ phép (휴가 신청 메시지)
                - Cuộc trò chuyện về phân công công việc (업무 분담 대화)
                - Cuộc hội thoại yêu cầu trợ giúp (도움 요청 대화)

            ## Nhiệm vụ của bạn: Tạo ra các nội dung tương tự với các chủ đề đã đưa ra
            ## Ví dụ minh họa:
                Thẻ đăng ký người nước ngoài:
                외국인 등록증
                외국인 등록번호: 990707-6******
                성명: 휘엔
                국가: 베트남
                체류자격: 비전문취업(E-9)
                발급일자: 2024.5.10.
                서울출입국·외국인청장

            ## Ví dụ minh họa 2:
                Bảng lương:

                성명: 마두
                부서: 생산부
                은행: 한국은행
                계좌 번호: 123-4567-89890

                지급 내역
                기본급: 2,060,740
                휴일 근로 수당: 179,640
                야간 근로 수당: 266,220
                상여금: 100,000
                식비
                지급액 합계: 2,606,600

                공제 내역
                소득세: 21,450
                주민세: 2,140
                국민연금: 92,730
                건강보험: 73,050
                고용보험: 18,540
                공제액 합계: 207,910

            ## Ví dụ minh họa 3:
                Thẻ đăng ký người nước ngoài:


                외국인 등록증
                외국인 등록번호: 990707-6******
                성명: 휘엔
                국가: 베트남
                체류자격: 비전문취업(E-9)
                발급일자: 2024. 5. 10.
                서울출입국·외국인청장

            ## Chú ý: 
            - Bạn hãy tạo {number} đoạn theo mẫu trên
            - Sau khi tạo xong, hãy kiểm tra lại xem có bị trùng lặp không, nếu có hãy tạo lại.
            - Hãy đánh số thứ tự bắt đầu từ 1.
            """
        )

    def call(self, number: int):
        response = self.llm.invoke(self.prompt_template.format(number=number))
        return response.content


if __name__ == "__main__":

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                   api_key=google_api_key,
                                   temperature=0                                   #    max_tokens=100000
                                   )
    # model = ChatOpenAI(model="gpt-4.1-mini",
    #                     api_key=openai_api_key,
    #                     temperature=0.2
    # )
    gen_question = GenQuestion(model)
    response = gen_question.call(10)
    print(response)
    with open("ccc.txt", "w", encoding="utf-8") as f:
        f.write(response)