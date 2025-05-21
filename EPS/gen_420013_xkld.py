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
            Bạn là chuyên gia về ngôn ngữ tiếng Hàn. Nhiệm vụ của bạn tạo ra câu hỏi trắc nghiệm cho người dùng chọn theo hướng dẫn dưới đây.
            ### Yêu cầu:
            Hãy tạo các câu hỏi tiếng Hàn dạng điền vào chỗ trống với nội dung và ngữ cảnh rõ ràng, kèm theo các phương án trả lời, đáp án đúng và giải thích chi tiết (bằng tiếng Anh và tiếng Việt). Mỗi câu hỏi phải mô tả một tình huống cụ thể, có ý nghĩa.
            Yêu cầu về nội dung:
            - Câu hỏi: là một đoạn văn gồm tối thiểu 4-5 câu văn, trong đó có 1-2 chỗ trống (______) cần điền từ/cụm từ/câu văn để hoàn thành ý nghĩa của đoạn văn.
            - Chỗ trống: cần được suy luận từ các câu văn còn lại trong đoạn văn.
            - Đoạn văn có thể là một câu chuyện, một tình huống cụ thể trong cuộc sống hàng ngày.
            - Mỗi câu văn phải có nghĩa rõ ràng, không mơ hồ, làm cho người đọc dễ hiểu và dễ hình dung.
            - Đoạn văn có thể có từ/cụm từ/câu văn cần điền là danh từ, động từ, tính từ, trạng từ, hoặc cụm danh từ.
            - Mỗi câu hỏi phải xây dựng một tình huống/ngữ cảnh cụ thể và rõ ràng (Cách xử lý khi bị thương nhẹ; Thái độ làm việc chuyên nghiệp.)
            - Chủ đề của các câu hỏi: chỉ trong phạm vi công việc, xuất khẩu lao động, không liên quan đến các chủ đề khác.
            - Câu hỏi cần kiểm tra hiểu biết về:
                + Từ vựng phù hợp với ngữ cảnh
                + Cấu trúc ngữ pháp chính xác
                + Cách diễn đạt tự nhiên trong tiếng Hàn
            
            ### Ví dụ:
            ## Câu hỏi: 자재 창고에는 많은 물품이 들어오고 나갑니다. 그래서 매일 물품을 세어서 맞는지 확인해야 합니다. 자재 창고 관리에서 가장 중요한 일은 ________.
            ## Câu trả lời: 	
            1. 물품을 판매하는 것입니다
            2. 온도를 유지하는 것입니다
            3. 재고를 파악하는 것입니다
            4. 자재를 운반하는 것입니다
            ## Đáp án đúng: 3
            ## Giải thích:
            Có rất nhiều hàng hóa ra vào kho. Vì vậy, bạn phải đếm số lượng hàng hóa mỗi ngày để đảm bảo chúng chính xác. Điều quan trọng nhất trong quản lý kho là ________.

            1. Bán hàng
            2. Duy trì nhiệt độ
            3. Nắm được hàng trong kho
            4. Vận chuyển vật liệu
            --------------------
            Phải đếm số lượng hàng hóa mỗi ngày để đảm bảo chúng chính xác => Phải nắm được hàng trong kho.
            ## Giải thích tiếng Anh:
            1. Sell the items
            2. Maintain the temperature
            3. Keep track of inventory
            4. Transport the materials
            --------------------
            There are many items coming in and going out of the warehouse. So, you have to count the items every day to make sure they are correct. The most important thing in warehouse management is to keep track of inventory.

            Must count the items every day to make sure they are correct => Must keep track of inventory.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 2 (trường hợp chỗ trống là câu văn):
            ## Câu hỏi: 좋은 직장 분위기를 만들기 위해서는 모든 사람의 노력이 필요합니다. 서로 웃는 얼굴로 이야기하는 것이 좋습니다. 그리고 상대방을 배려하는 마음을 갖는다면 ________.
            ## Câu trả lời:
            1. 즐거울 일이 없어질 것입니다
            2. 가깝게 지내지 않을 것입니다
            3. 화내는 일이 없어질 것입니다
            4. 분위기가 좋지 않을 것입니다
            ## Đáp án đúng: 3
            ## Giải thích:
            Việc tạo ra một môi trường làm việc tốt đòi hỏi sự nỗ lực của mọi người. Thật tốt khi nói chuyện với nhau với nụ cười trên môi. Và nếu bạn có tấm lòng bao dung với đối phương, ________.

            1. Sẽ không còn sự vui vẻ nữa
            2. Sẽ không thân thiết
            3. Sẽ không còn sự tức giận nữa
            4. Bầu không khí sẽ không tốt
            --------------------
            Nếu mọi người có tấm lòng bao dung đối phương thì sẽ không còn sự tức giận.
            ## Giải thích tiếng Anh:
            1. There will be no more fun things
            2. You won't be close
            3. There will be no more anger
            4. The atmosphere will be bad
            --------------------
            Creating a good work atmosphere requires everyone's effort. It's good to talk to each other with a smile. And if you have a considerate heart for the other person, there will be no more anger.

            If people have a considerate heart for the other person, there will be no more anger.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 3 (trường hợp chỗ trống là một cụm từ):
            ## Câu hỏi: 기계 작업을 할 때에는 안전 장갑과 보안경을 반드시 착용해야 합니다. 작업 전에는 기계를 점검하고, 기계가 작동 중일 때는 기계에 가까이 가지 않도록 주의해야 합니다. ________ 기계의 전원을 반드시 꺼야 안전사고를 예방할 수 있습니다.
            ## Câu trả lời:
            1. 작업이 끝나면
            2. 작업이 필요하면
            3. 기계가 설치되면
            4. 기계가 작동하면
            ## Đáp án đúng: 1
            ## Giải thích:
            Phải đeo găng tay và kính an toàn khi làm việc với máy móc. Trước khi làm việc, hãy kiểm tra máy và cẩn thận không đến gần máy khi máy đang hoạt động. ________, hãy nhớ tắt máy để tránh xảy ra tai nạn về an toàn.

            1. Khi công việc hoàn thành
            2. Nếu cần làm việc
            3. Sau khi máy được lắp đặt
            4. Khi máy đang hoạt động
            --------------------
            Việc tắt máy để tránh xảy ra tai nạn về an toàn cần được thực hiện khi công việc đã hoàn thành.
            ## Giải thích tiếng Anh:
            1. When you finish working
            2. If work is required
            3. When the machine is installed
            4. When the machine is in operation
            --------------------
            When working on a machine, you must wear safety gloves and safety glasses. Check the machine before working, and be careful not to get close to the machine while it is operating. When you finish working, you must turn off the power to the machine to prevent safety accidents.

            Turning off the machine to prevent safety accidents should be done when the work is finished.

            ## Chú ý: 
            - Bạn hãy tạo đủ {number} câu hỏi trắc nghiệm, không nói gì thêm. Trong đó
                + Gồm {number}/3 câu hỏi có chỗ trống là một từ.
                + Gồm {number}/3 câu hỏi có chỗ trống là một cụm từ.
                + Gồm {number}/3 câu hỏi có chỗ trống là một câu văn.
            - Sau khi tạo xong, hãy kiểm tra lại xem có câu hỏi nào bị trùng lặp không, nếu có hãy tạo lại câu hỏi đó.
            - Các câu hỏi và câu trả lời chỉ được chứa tiếng Hàn.
            - Hãy đánh số thứ tự bắt đầu từ 41.
            """
        )
    
    def call(self, number: int):
        response = self.llm.invoke(self.prompt_template.format(number=number))
        return response.content
    
if __name__ == "__main__":

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                   api_key=google_api_key, 
                                   temperature=0.5,
                                #    max_tokens=1000000
    )
    # model = ChatOpenAI(model="gpt-4.1",
    #                     api_key=openai_api_key,
    #                     temperature=0.5
    # )
    gen_question = GenQuestion(model)
    response = gen_question.call(10)
    print(response)
    with open("output_420013_xkld.txt", "w", encoding="utf-8") as f:
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