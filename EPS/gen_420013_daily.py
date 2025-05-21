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
            Hãy tạo các câu hỏi tiếng Hàn dạng điền vào chỗ trống với nội dung và ngữ cảnh rõ ràng, kèm theo các phương án trả lời, đáp án đúng và giải thích chi tiết. Mỗi câu hỏi phải mô tả một tình huống cụ thể, có ý nghĩa.
            Yêu cầu về nội dung:
            - Câu hỏi: là một đoạn văn gồm tối thiểu 4-5 câu văn, trong đó có 1-2 chỗ trống (______) cần điền từ/cụm từ/câu văn để hoàn thành ý nghĩa của đoạn văn.
            - Chỗ trống: cần được suy luận từ các câu văn còn lại trong đoạn văn.
            - Đoạn văn có thể là một câu chuyện, một tình huống cụ thể trong cuộc sống hàng ngày.
            - Mỗi câu văn phải có nghĩa rõ ràng, không mơ hồ, làm cho người đọc dễ hiểu và dễ hình dung.
            - Đoạn văn có thể có từ/cụm từ/câu văn cần điền là danh từ, động từ, tính từ, trạng từ, hoặc cụm danh từ.
            - Mỗi câu hỏi phải xây dựng một tình huống/ngữ cảnh cụ thể và rõ ràng (Nghe nhạc cả buổi tối; Thay ảnh đại diện mới; Viết nhật ký mỗi tối; Gặp người nổi tiếng bất ngờ; Tập viết blog cá nhân)
            - Chủ đề của các câu hỏi: chỉ trong phạm vi cuộc sống hàng ngày, không liên quan đến các chủ đề khác.
            - Câu hỏi cần kiểm tra hiểu biết về:
                + Từ vựng phù hợp với ngữ cảnh
                + Cấu trúc ngữ pháp chính xác
                + Cách diễn đạt tự nhiên trong tiếng Hàn
            
            ### Ví dụ:
            ## Câu hỏi: 제 고향은 ________. 저는 한국에 와서 처음 겨울을 경험했습니다. 너무 추워서 깜짝 놀랐습니다.
            ## Câu trả lời: 	
            1. 바람이 많이 붑니다
            2. 비가 자주 옵니다
            3. 날씨가 항상 따뜻합니다
            4. 날씨가 너무 춥습니다
            ## Đáp án đúng: 3
            ## Giải thích:
            Ở quê tôi ________. Tôi đã lần đầu tiên trải nghiệm mùa đông khi đến Hàn Quốc. Lạnh quá nên tôi đã rất bất ngờ.

            1. Trời có gió rất lớn
            2. Trời mưa thường xuyên
            3. Thời tiết luôn ấm áp
            4. Thời tiết quá lạnh
            --------------------
            Bất ngờ trước cái lạnh của mùa đông Hàn Quốc => Ở quê thời tiết luôn ấm áp.
            ## Giải thích tiếng Anh:
            1. It's very windy
            2. It rains often
            3. The weather is always warm
            4. The weather is too cold
            --------------------
            The weather is always warm in my hometown. I experienced winter for the first time when I came to Korea. I was surprised because it was so cold.

            This person was surprised by the coldness of Korean winter => The weather in his/her hometown is always warm.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 2:
            ## Câu hỏi: 어제 다리를 다쳤습니다. 걸으면 다리가 너무 아픕니다. 그래서 병원에 가서 치료를 ________.
            ## Câu trả lời:
            1. 보내려고 합니다
            2. 시키려고 합니다
            3. 받으려고 합니다
            4. 만들려고 합니다
            ## Đáp án đúng: 3
            ## Giải thích:
            Hôm qua tôi bị đau chân. Nếu đi bộ thì chân tôi đau lắm. Vì vậy, tôi ____ đến bệnh viện để ____ điều trị.

            1. Định - gửi
            2. Định - đặt
            3. Định - nhận
            4. Định - làm
            --------------------
            치료를 받다: Nhận điều trị.
            ## Giải thích tiếng Anh:
            1. I'm going to send
            2. I'm going to order
            3. I'm going to get
            4. I'm going to make
            --------------------
            I hurt my leg yesterday. My leg hurts so much when I walk. So I'm going to the hospital to get treatment.

            <g>치료를 받다</g>: To get treatment.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ## Ví dụ 3:
            ## Câu hỏi: 나는 지난 주말에 이사를 했습니다. 새집 근처에는 공원이 있어서 산책을 할 수 있고 마트도 가깝습니다. 새집이 정말 ________.
            ## Câu trả lời:
            1. 마음에 듭니다
            2. 넓고 깨끗합니다
            3. 교통이 편리합니다
            4. 보증금이 저렴합니다
            ## Đáp án đúng: 1
            ## Giải thích:
            Tôi đã chuyển nhà vào cuối tuần trước. Có một công viên gần nhà mới, nơi tôi có thể đi dạo và cũng có một siêu thị ở gần đó. ________ ngôi nhà mới của tôi.

            1. Tôi thực sự thích
            2. Rộng rãi và sạch sẽ
            3. Giao thông thuận tiện
            4. Tiền đặt cọc thấp
            --------------------
            Các đặc điểm được đề cập ở câu thứ hai khiến người này hài lòng với căn nhà mới.
            ## Giải thích tiếng Anh:
            1. I really like
            2. It is spacious and clean
            3. Transportation is convenient
            4. The deposit is low
            --------------------
            I moved last weekend. There is a park near my new house where I can go for a walk and there is a supermarket nearby. I really like my new house.

            The features mentioned in the second sentence make this person like his/her new house.

            ## Chú ý: 
            - Bạn hãy tạo đủ {number} câu hỏi trắc nghiệm, không nói gì thêm.
            - Sau khi tạo xong, hãy kiểm tra lại xem có câu hỏi nào bị trùng lặp không, nếu có hãy tạo lại câu hỏi đó.
            - Các câu hỏi và câu trả lời chỉ được chứa tiếng Hàn.
            - Hãy đánh số thứ tự bắt đầu từ 51.
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
    with open("output_420013_daily.txt", "w", encoding="utf-8") as f:
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