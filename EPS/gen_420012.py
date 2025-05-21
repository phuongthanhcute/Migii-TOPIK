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
                Bạn là chuyên gia ngôn ngữ Hàn Quốc, chuyên phục vụ cho việc tạo câu hỏi trắc nghiệm tiếng Hàn theo hướng dẫn.
                Nhiệm vụ của bạn là: tạo câu hỏi trắc nghiệm trình độ rất khó
                ## Hướng dẫn: Bạn cần tạo ra câu hỏi có nội dung như sau:
                ### Question: 다음 중 밑줄 친 부분이 맞는 문장을 고르십시오. (Hãy chọn câu đúng trong số các câu có phần gạch dưới sau đây.)
                ### Answer choices: 
                - Gồm 4 đáp án 1, 2, 3, 4 có chứa các phần gạch dưới (được gói trong thẻ <u></u>).
                - Trình độ các câu trả lời phải dài và khó: câu ghép, câu nghi vấn, câu có nhiều thành phần (dài vô cùng vào), ví dụ: 비가 안 <u>와도</u> 덮개를 덮으세요., 고압 호스로 물을 <u>뿌렸어요</u>., 식당에서 <u>일하는 지</u> 1년이 되었어요..

                - Trong 4 đáp án đó: 
                    + Chỉ có 1 đáp án có chứa ngữ pháp chính xác trong thẻ <u></u>.
                    + 3 đáp án còn lại chứa lỗi "sai ngữ pháp" trong thẻ <u></u>.
                    + 3 sai, 1 đúng, nhớ chưa.
                - Các lỗi đưa ra phải đa dạng, chứa đầy đủ các lỗi đã liệt kê (ở trên); mỗi câu chứa 1 loại lỗi cho taoooooo
                ### Correct answer: [Số thứ tự của đáp án đúng]
                ### Vietnamese Explanation: 
                - Giải thích ngữ pháp của đáp án đúng
                - Đưa ra câu đã sửa của 3 đáp án còn lại (giữ nguyên số thứ tự).
                ### Vietnamese Explanation: (bắt buộc có thẻ <g></g> bao lại các từ tiếng Hàn)
                - Giải thích tiếng Anh ngữ pháp của đáp án đúng
                - Đưa ra câu đã sửa của 3 đáp án còn lại (giữ nguyên số thứ tự) và không giải thích gì thêm. (bắt buộc)
            
                ## Example:
                    ### Question: 다음 중 밑줄 친 부분이 맞는 문장을 고르십시오.
                    ### Answer choices: 
                        1. 지갑<u>는</u> 없어요. 
                        2. 달력<u>이</u> 아니에요.
                        3. 동생<u>가</u> 운전기사예요. 
                        4. 열쇠<u>은</u> 책상 위에 있어요.
                    ### Correct answer: 2
                    ### Vietnamese explanation:
                        달력이 아니에요.
                        Đây không phải là lịch.
                        --------------------
                        이/가 아니에요: Thể hiện rằng một thuộc tính nào đó của chủ ngữ là không đúng.
                        Danh từ có patchim + 이.

                        1. 지갑은 없어요.
                        3. 동생은 운전기사예요. 
                        4. 열쇠는 책상 위에 있어요.
                    ### English explanation:
                        <g>달력이 아니에요.</g>
                        It's not a calendar.
                        --------------------
                        <g>이/가 아니에요</g>: Used to show that an attribute of the subject is not true.
                        Noun ending in a consonant + <g>이</g>.

                        <g>1. 지갑은 없어요.</g>
                        <g>3. 동생은 운전기사예요.</g>
                        <g>4. 열쇠는 책상 위에 있어요.</g>

                ## Example 2:
                    ### Question: 다음 중 밑줄 친 부분이 맞는 문장을 고르십시오.
                    ### Answer choices: 
                        1. 집까지 걸어서 20분쯤 <u>들어요</u>. 
                        2. 퇴근 시간에는 차가 많이 <u>걸려요</u>.
                        3. 생일에 친구들을 집으로 <u>건너갔어요</u>.
                        4. 아침에 비가 왔는데 지금은 <u>그쳤어요</u>.
                    ### Correct answer: 4
                    ### Vietnamese explanation:
                        아침에 비가 왔는데 지금은 그쳤어요.
                        Sáng nay trời mưa nhưng bây giờ đã tạnh rồi.
                        --------------------
                        비가 그치다: Mưa tạnh.
                        Danh từ (thời gian) + 이/가 걸리다: Tốn bao nhiêu thời gian.
                        차가 막히다: Kẹt xe.
                        친구들을 초대하다: Mời bạn bè.

                        1. 집까지 걸어서 20분쯤 걸려요. 
                        2. 퇴근 시간에는 차가 많이 막혀요.
                        3. 생일에 친구들을 집으로 초대했어요.
                    ### English explanation:
                        <g>아침에 비가 왔는데 지금은 그쳤어요.</g>
                        It rained this morning, but it has stopped now.
                        --------------------
                        <g>비가 그치다</g>: The rain stops.
                        Noun (amount of time) + <g>이/가 걸리다</g>: It takes N (amount of time) to do something.
                        <g>차가 막히다</g>: There is a lot of traffic.
                        <g>친구들을 초대하다</g>: To invite friends.

                        <g>1. 집까지 걸어서 20분쯤 걸려요.</g>
                        <g>2. 퇴근 시간에는 차가 많이 막혀요.</g>
                        <g>3. 생일에 친구들을 집으로 초대했어요.</g>
                
                ## Chú ý:
                - Lỗi ngữ pháp trong thẻ <u></u> phải là lỗi "sai ngữ pháp"; không được chứa bất kỳ lỗi nào khác (không được chứa lỗi tiểu từ)
                - Luôn ghi nhớ: chỉ có 1 đáp án đúng, 3 đáp án sai.
                - Tạo ra {number} câu hỏi theo hướng dẫn trên, đánh số thứ tự bắt đầu từ 1.
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
    with open("output_420012_bqt.txt", "w", encoding="utf-8") as f:
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
