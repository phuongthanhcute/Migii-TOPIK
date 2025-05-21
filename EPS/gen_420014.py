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
            Bạn là một chuyên gia ngôn ngữ Hàn Quốc. Nhiệm vụ cùa bạn là tạo câu hỏi trắc nghiệm phục vụ kỳ thi EPS theo hướng dẫn dưới đây:
            Hướng dẫn tạo câu hỏi trắc nghiệm:
            ## Đầu vào: Tài liệu đầu vào:
                - Tài liệu visa (비자 서류)
                - Giấy chứng nhận y tế hoặc đơn thuốc (의료 증명서/처방전)
                - Sao kê ngân hàng (은행 거래 내역서)
                - Tài liệu bảo hiểm (보험 서류)

                2. **Thông tin hàng ngày**
                - Sự kiện giảm giá tại cửa hàng (마트 할인 행사)
                - Vé phương tiện giao thông công cộng (대중교통 승차권)
                - Thông báo sự kiện văn hóa (문화 행사 안내)
                - Hướng dẫn an toàn (안전 수칙)
                - Thông báo chương trình đào tạo (교육 안내)
                - Thông báo tuyển dụng (채용 공고)

                3. **Giao tiếp cá nhân**
                - Đoạn hội thoại tin nhắn giữa đồng nghiệp (동료 간의 문자 대화)
                - Tin nhắn liên quan đến công việc (업무 관련 메시지)
                - Cuộc trò chuyện thường ngày (일상 대화)
                - Tin nhắn về thủ tục hành chính (행정 절차 관련 메시지)
                ...

            ## Nhiệm vụ của bạn: Tạo ra các nội dung sau đây dựa trên tài liệu đầu vào {theme}
                ### Danh sách đáp án: 
                    - Gồm 4 đáp án 1, 2, 3, 4 gồm các câu văn miêu tả/trần thuật về nội dung xuất hiện trong tài liệu.
                    - Chỉ có một đáp án đúng với nội dung của tài liệu đầu vào.
                    - 3 đáp án còn lại mô tả sai về tài liệu đào tạo
                ### Đáp án đúng; [Số thứ tự của đáp án đúng]
                ### Giải thích tiếng Việt:
                    - Giải thích ý nghĩa của các trường thông tin được nêu ra trong tài liệu.
                    - Giải thích nghĩa tiếng Việt của 4 đáp án.
                ### Giải thích tiếng Anh:
                    - Giải thích ý nghĩa của các trường thông tin được nêu ra trong tài liệu.
                    - Giải thích nghĩa tiếng Anh của 4 đáp án.
                    - Thêm thẻ <g></g> vào tất cả các ký tự tiếng Hàn ở phần giải thích tiếng Anh.
            ## Ví dụ minh họa:
            ### Tài liệu đầu vào:
                Thẻ đăng ký người nước ngoài:
                외국인 등록증
                외국인 등록번호: 990707-6******
                성명: 휘엔
                국가: 베트남
                체류자격: 비전문취업(E-9)
                발급일자: 2024.5.10.
                서울출입국·외국인청장
            ### Danh sách đáp án:
                1. 베트남 사람입니다.
                2. 생일은 5월 10일입니다.
                3. 7월에 외국인 등록증을 받았습니다.
                4. 서울출입국 · 외국인청에서 일합니다.
            ### Đáp án đúng: 1
            ### Giải thích tiếng Việt:
                외국인 등록증 (Thẻ đăng ký người nước ngoài)
                외국인 등록번호: 990707-6****** (Số đăng ký người nước ngoài: 990707-6******)
                성명: 휘엔 (Tên: Huyền)
                국가: 베트남 (Quốc gia: Việt Nam)
                체류자격: 비전문취업(E-9) (Tư cách cư trú: Lao động phổ thông (E-9))
                발급일자: 2024.5.10. (Ngày cấp: 10/05/2024)
                서울출입국·외국인청장 (Giám đốc Văn phòng xuất nhập cảnh Seoul)

                1. Đây là người Việt Nam.
                2. Ngày sinh của người này là ngày 10 tháng 5.
                3. Người này đã nhận được thẻ đăng ký người nước ngoài vào tháng 7.
                4. Người này làm việc tại Văn phòng xuất nhập cảnh Seoul.
                --------------------
                국가: 베트남 (Quốc gia: Việt Nam) => Đây là người Việt Nam.
            ### Giải thích tiếng Anh: (chứa thẻ <g></g> với các nội dung tiếng Hàn)
                <g>외국인 등록증</g> (Alien Registration Card)
                <g>외국인 등록번호: 990707-6******</g> (Alien registration number: 990707-6******)
                <g>성명: 휘엔</g> (Name: Huyen)
                <g>국가: 베트남</g> (Country: Vietnam)
                <g>체류자격: 비전문취업(E-9)</g> (Status of residence: Non-professional employment (E-9))
                <g>발급일자: 2024.5.10.</g> (Issue date: May 10th, 2024)
                <g>서울출입국·외국인청장</g> (Seoul Immigration Office Director)

                1. This person is Vietnamese.
                2. This person's birthday is May 10th.
                3. This person received her alien registration card in July.
                4. This person works at the Seoul Immigration Office.
                --------------------
                <g>국가: 베트남</g> (Country: Vietnam) => This person is Vietnamese.

            ## Chú ý: 
            - Bạn hãy tạo {number} đoạn theo mẫu trên
            - Sau khi tạo xong, hãy kiểm tra lại xem có bị trùng lặp không, nếu có hãy tạo lại.
            - Hãy đánh số thứ tự bắt đầu từ 1.
            """
        )
    
    def __call__(self, *,number: int, theme: str) -> str:
        response = self.llm.invoke(self.prompt_template.format(number=number, theme=theme))
        return response.content
    
if __name__ == "__main__":

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                   api_key=google_api_key, 
                                   temperature=0.5,
                                #    max_tokens=100000
    )
    # model = ChatOpenAI(model="gpt-4.1",
    #                     api_key=openai_api_key,
    #                     temperature=0.5
    # )
    gen_question = GenQuestion(model)
    while True:
        print("Input:")
        user_input = sys.stdin.read().strip()
        response = gen_question(number=1, theme=user_input)
        print(response)
        with open("output_420014_ques.txt", "w", encoding="utf-8") as f:
            f.write(response + "\n")
    # while True:
    #     print("Input: (ctrl D để dừng):")
    #     article = sys.stdin.read().strip()
        
    #     if article.lower() == 'x':
    #         break
        
    #     response = gen_question(article)
    #     print("-" * 80)
    #     print(response)
    #     print("-" * 80)