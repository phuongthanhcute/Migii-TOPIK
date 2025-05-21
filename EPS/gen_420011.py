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
            Bạn là chuyên gia về ngôn ngữ tiếng Hàn. Nhiệm vụ của bạn tạo ra câu hỏi trắc nghiệm cho người dùng chọn về từ trái nghĩa, và giải thích nghĩa của từ đó.
            ### Yêu cầu:
            - Đáp án sẽ là 4 lựa chọn 1, 2, 3, 4. Trong đó chỉ có 1 đáp án đúng. 
            - Trình độ từ vựng: trung cấp.
            - Chủ đề từ vựng: thuộc nhiều chủ đề khác nhau như văn hóa Hàn quốc (lễ Tết,...), xã hội, giáo dục, sức khỏe, công việc, thể thao, du lịch, ẩm thực Hàn quốc, thời trang, công nghệ, giải trí,...
            - Đa dạng loại từ vựng: bắt buộc phải là danh từ
            - Các câu hỏi được tạo ra cho bài thi tiếng Hàn thuộc chương trình cấp phép lao động EPS (Employment Permit System).
            
            ### Ví dụ:
            ## Ví dụ 1:
            ## Câu hỏi: 자유
            ## Câu trả lời:
            1. 평화
            2. 억압
            3. 권리
            4. 행복
            ## Đáp án đúng: 2
            ## Giải thích:
            Tự do
            1. Hòa bình
            2. Áp bức
            3. Quyền lợi
            4. Hạnh phúc
            ## Giải thích tiếng Anh:
            Freedom
            1. Peace
            2. Oppression
            3. Rights
            4. Happiness
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 2:
            ## Câu hỏi: 전쟁
            ## Câu trả lời:
            1. 평화
            2. 병사
            3. 무기
            4. 공격
            ## Đáp án đúng: 1
            ## Giải thích:
            Chiến tranh
            1. Hòa bình (trái nghĩa với chiến tranh)
            2. Binh lính
            3. Vũ khí
            4. Tấn công
            ## Giải thích tiếng Anh:
            War
            1. Peace (opposite of war)
            2. Soldiers
            3. Weapons
            4. Attack
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
    response = gen_question.call(90)
    print(response)
    with open("output_420011_noun.txt", "w", encoding="utf-8") as f:
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