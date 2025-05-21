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
            Bạn là chuyên gia về ngôn ngữ tiếng Hàn. Nhiệm vụ của bạn tạo ra câu hỏi trắc nghiệm cho người dùng và giải thích theo yêu cầu sau:

            **Yêu cầu:** Nội dung câu hỏi luôn luôn là câu hỏi về chủ đề của một đoạn hội thoại cho trước gồm 7 phần như sau:
                ### Text Audio: Đoạn hội thoại gồm 2 câu, bao gồm 1 nam và 1 nữ cùng về một chủ đề nào đó.
                ### Vietnamese: Bản dịch tiếng Việt của đoạn hội thoại trên.
                ### English: Bản dịch tiếng Anh của đoạn hội thoại trên.
                ### Answer: Bao gồm 4 đáp án (tiếng Hàn) 1, 2, 3, 4. Trong đó chỉ có 1 đáp án là nội dung của đoạn hội thoại trên.
                ### Correct Answer: [Số thứ tự của đáp án đúng]
                ### Vietnamese Explain: Dịch tiếng Việt 4 đáp án và giải thích.
                ### English Explain: Dịch tiếng Anh 4 đáp án và giải thích bằng tiếng Hàn.

                - Các câu hỏi được tạo ra cho bài thi tiếng Hàn thuộc chương trình cấp phép lao động EPS (Employment Permit System).
                - Chủ đề của đoạn hội thoại: những sự kiện xảy ra trong cuộc sống hàng ngày, bao gồm: Bảo trì;Dụng cụ sửa chữa;Thiết bị hỏng;Lương
                - Tuân thủ đúng format theo ví dụ bên dưới.
            ** Ví dụ minh họa 1 **
                ### Text Audio: 
                    여자: 안녕하세요? 저는 네팔 사람입니다.
                    남자: 안녕하세요? 저는 몽골 사람입니다.
                ### Vietnamese: 
                    Nữ: Xin chào? Tôi là người Nepal. 
                    Nam: Xin chào? Tôi là người Mông Cổ.
                ### English: 
                    Woman: Hello? I am from Nepal. 
                    Man: Hello? I am from Mongolia.
                ### Answer: 
                    1. 이름 
                    2. 나이
                    3. 국적
                    4. 성별
                ### Correct Answer: 3
                ### Vietnamese Explain: 
                    1. Tên 
                    2. Tuổi 
                    3. Quốc tịch 
                    4. Giới tính
                    ------------------
                    네팔 사람, 몽골 사람 => 국적.
                ### English Explain: 
                    1. Name 
                    2. Age 
                    3. Nationality 
                    4. Gender
                    ------------------
                    <g>네팔 사람, 몽골 사람 => 국적.</g>
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ** Ví dụ minh họa 2 **
                ### Text Audio: 
                    남자: 이 티셔츠는 오만 원이에요.
                    여자: 티셔츠가 비싸네요. 좀 깎아 주세요.
                ### Vietnamese: 
                    Nam: Áo thun này giá năm mươi ngàn won. 
                    Nữ: Áo thun đắt quá. Hãy giảm giá cho tôi chút.
                ### English: 
                    Man: This T-shirt is fifty thousand won. 
                    Woman: The T-shirt is expensive. Please give me a discount.
                ### Answer: 
                    1. 가격
                    2. 날씨
                    3. 장소
                    4. 직업
                ### Correct Answer: 1
                ### Vietnamese Explain: 
                    1. Giá cả 
                    2. Thời tiết 
                    3. Địa điểm 
                    4. Nghề nghiệp
                    ------------------
                    오만 원, 비싸다, 깎다 => 가격.
                ### English Explain: 
                    1. Price 
                    2. Weather 
                    3. Place 
                    4. Job
                    ------------------
                    <g>오만 원, 비싸다, 깎다 => 가격.</g>

            ## Chú ý: 
            - Bạn hãy tạo {number} câu hỏi trắc nghiệm.
            - Sau khi tạo xong, hãy kiểm tra lại xem có câu hỏi nào bị trùng lặp không, nếu có hãy tạo lại câu hỏi đó.
            - Các câu hỏi và câu trả lời chỉ được chứa tiếng Hàn.
            - Hãy đánh số thứ tự bắt đầu từ 71.
            """
        )
    
    def call(self, number: int):
        response = self.llm.invoke(self.prompt_template.format(number=number))
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
    response = gen_question.call(10)
    print(response)
    with open("output_420009.txt", "w", encoding="utf-8") as f:
        f.write(response)
    # while True:
    #     print("Input: (ctrl D để dừng):")
    #     article = sys.stdin.read().strip()
        
    #     if article.lower() == 'x':
    #         break
        
    #     response = gen_question(article)
    #     print("-" * 80)
    #     print(response)
    #     print("-" * 80)