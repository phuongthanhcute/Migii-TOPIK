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
            ### Yêu cầu:
            ### Câu hỏi: Là câu văn miêu tả đồ vật/sự kiện/... nào đó có nội dung như sau:
            - "Đây là <cái gì đó>" (40 câu)
            - "Đang làm <cái gì đó>" (40 câu)
            
            ### Câu trả lời: Gồm 4 đáp án (bắt buộc là tiếng Việt) 1, 2, 3, 4. Trong đó chỉ có 1 đáp án đúng với nội dung được miêu tả trong câu hỏi.
            ### Đáp án đúng: [Số thứ tự của đáp án đúng]
            ### Giải thích tiếng Việt: Dịch nghĩa câu hỏi (chỉ dịch câu hỏi) sang tiếng Việt. (Nếu ẩn chủ ngữ thì tự thêm vào)
            ### Giải thích tiếng Anh: Dịch nghĩa câu hỏi (chỉ dịch câu hỏi) sang tiếng Anh. (Nếu ẩn chủ ngữ thì tự thêm vào)
            - Các câu hỏi được tạo ra cho bài thi tiếng Hàn thuộc chương trình cấp phép lao động EPS (Employment Permit System).
            
            ### Ví dụ:
            ## Câu hỏi: 우산입니다.
            ## Câu trả lời:
            1. Cái bàn
            2. Cái ô 
            3. Cái ghế
            4. Cái áo mưa
            ## Đáp án đúng: 2
            ## Giải thích: Đây là một cái ô.
            ## Giải thích tiếng Anh: This is an umbrella.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 2:
            ## Câu hỏi: 요리를 하고 있습니다.
            ## Câu trả lời:
            1. Nấu ăn
            2. Đi bộ
            3. Rửa bát
            4. Học bài
            ## Đáp án đúng: 1
            ## Giải thích: Người này đang nấu ăn.
            ## Giải thích tiếng Anh: This person is cooking.
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
    response = gen_question.call(80)
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