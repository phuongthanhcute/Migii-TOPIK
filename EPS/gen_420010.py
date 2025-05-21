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
            Bạn là chuyên gia về ngôn ngữ tiếng Hàn. Nhiệm vụ của bạn tạo ra câu hỏi trắc nghiệm cho người dùng chọn về từ đồng nghĩa, và giải thích nghĩa của từ đó.
            ### Yêu cầu:
            - Đáp án sẽ là 4 lựa chọn 1, 2, 3, 4. Trong đó chỉ có 1 đáp án đúng. 
            - Trình độ từ vựng: trung cấp.
            - Chủ đề từ vựng: thuộc nhiều chủ đề khác nhau như văn hóa Hàn quốc (lễ Tết,...), xã hội, giáo dục, sức khỏe, công việc, thể thao, du lịch, ẩm thực Hàn quốc, thời trang, công nghệ, giải trí,...
            - Chủ đề bắt buộc: xuất khẩu lao động.
            - Đa dạng loại từ vựng:  bắt buộc là động từ
            - Các câu hỏi được tạo ra cho bài thi tiếng Hàn thuộc chương trình cấp phép lao động EPS (Employment Permit System).
            
            ### Ví dụ:
            ## Câu hỏi: 자르다
            ## Câu trả lời:
            1. 조립하다
            2. 혼합하다 
            3. 절단하다
            4. 조절하다
            ## Đáp án đúng: 3
            ## Giải thích:
            Cắt

            1. Lắp ráp
            2. Trộn
            3. Cắt
            4. Điều tiết
            --------------------
            자르다 = 절단하다.

            ## Giải thích tiếng Anh:
            To cut

            1. To assemble
            2. To mix
            3. To cut
            4. To control
            -------------------
            <g>자르다 = 절단하다.</g>
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 2:
            ## Câu hỏi: 강하다
            ## Câu trả lời:
            1. 튼튼하다
            2. 당기다
            3. 금지하다
            4. 낭비하다
            ## Đáp án đúng: 1
            ## Giải thích:
            Mạnh

            1. Chắc chắn
            2. Kéo
            3. Cấm
            4. Lãng phí
            ## Giải thích tiếng Anh:
            To be strong 

            1. To be strong
            2. To pull
            3. To ban
            4. To waste
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            ## Chú ý: 
            - Bạn hãy tạo {number} câu hỏi trắc nghiệm.
            - Sau khi tạo xong, hãy kiểm tra lại xem có câu hỏi nào bị trùng lặp không, nếu có hãy tạo lại câu hỏi đó.
            - Các câu hỏi và câu trả lời chỉ được chứa tiếng Hàn.
            - Hãy đánh số thứ tự bắt đầu từ 1.
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
    response = gen_question.call(30)
    print(response)
    with open("output_420010_more.txt", "w", encoding="utf-8") as f:
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