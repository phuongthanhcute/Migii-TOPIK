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

            **Yêu cầu:** Nội dung câu hỏi luôn luôn là "Chọn đáp án chứa đúng thông tin được nhắc đến trong đoạn hội thoại" cho trước gồm 7 phần như sau:
                ### Text Audio: Đoạn hội thoại gồm 3-4 câu, xen kẽ giữa 1 nam và 1 nữ cùng về một chủ đề nào đó.
                ### Vietnamese: Bản dịch tiếng Việt của đoạn hội thoại trên.
                ### English: Bản dịch tiếng Anh của đoạn hội thoại trên.
                ### Answer: Bao gồm 4 đáp án (tiếng Hàn) 1, 2, 3, 4. là các câu trần thuật, mô tả một sự việc nào đó. Trong đó chỉ có 1 đáp án có chứa thông tin được nhắc đến trong đoạn hội thoại trên.
                ### Correct Answer: [Số thứ tự của đáp án đúng]
                ### Vietnamese Explain: Dịch tiếng Việt 4 đáp án và giải thích.
                ### English Explain: Dịch tiếng Anh 4 đáp án và giải thích bằng tiếng Hàn.

                - Các câu hỏi được tạo ra cho bài thi tiếng Hàn thuộc chương trình cấp phép lao động EPS (Employment Permit System).
                - Chủ đề của đoạn hội thoại: những sự kiện xảy ra trong cuộc sống hàng ngày, bao gồm: Thích nghi văn hóa nước ngoài;Quyền lợi người lao động xuất khẩu.
                
                Tuân thủ theo format của ví dụ bên dưới.
                ** Ví dụ minh họa 1 **
                ### Text Audio: 
                    남자: 자야 씨, 주말에 뭐 했어요?
                    여자: 저는 콘서트를 봤어요. 마두 씨는요?
                    남자: 저는 친구를 만났어요. 친구랑 점심을 먹고 공원에 가서 배드민턴을 쳤어요.
                ### Vietnamese: 
                    Nam: Jaya, cuối tuần cô đã làm gì?
                    Nữ: Tôi đã đi xem buổi hòa nhạc. Còn Madu thì sao?
                    Nam: Tôi đã gặp bạn. Chúng tôi ăn trưa rồi ra công viên chơi cầu lông.
                ### English: 
                    Man: Jaya, what did you do over the weekend?
                    Woman: I watched a concert. What about you, Madu?
                    Man: I met my friend. We had lunch together and then went to the park to play badminton.
                ### Answer: 
                    1. 남자는 친구를 만났습니다.
                    2. 남자는 콘서트를 봤습니다.  
                    3. 여자는 남자와 배드민턴을 쳤습니다. 
                    4. 두 사람은 주말에 같이 밥을 먹었습니다.
                ### Correct Answer: 1
                ### Vietnamese Explain: 
                    1. Người đàn ông đã gặp bạn.
                    2. Người đàn ông đã xem một buổi hòa nhạc.
                    3. Người phụ nữ đã chơi cầu lông với người đàn ông.
                    4. Hai người đã cùng nhau ăn cơm vào cuối tuần.
                    --------------------
                    남자: 저는 친구를 만났어요. (Nam: Tôi đã gặp bạn.)
                ### English Explain: 
                    1. The man met his friend.
                    2. The man watched a concert.
                    3. The woman played badminton with the man.
                    4. The two people ate together on the weekend.
                    --------------------
                    <g>남자: 저는 친구를 만났어요.</g> (Man: I met my friend.)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ** Ví dụ minh họa 2 **
                ### Text Audio: 
                    여자: 유수프 씨, 여기서 요리하면 안 돼요.
                    남자: 어, 휴게실에서 요리하면 안 되는지 몰랐어요.
                    여자: 요리는 조리실에서 해야 돼요. 아홉 시까지 이용할 수 있어요. 아, 요리한 음식을 휴게실에서 먹는 건 괜찮아요.
                    남자: 아, 그래요? 알려주셔서 감사합니다.
                ### Vietnamese: 
                    Nữ: Yusuf, anh không được nấu ăn ở đây.
                    Nam: Ồ, tôi không biết là không được nấu ăn trong phòng giải lao.
                    Nữ: Anh phải nấu ăn trong phòng nấu ăn. Phòng nấu ăn có thể sử dụng đến 9 giờ. À, nhưng ăn thức ăn đã nấu trong phòng giải lao thì không sao.
                    Nam: Ồ, vậy à? Cảm ơn vì đã nói cho tôi biết.
                ### English: 
                    Woman: Yusuf, you're not allowed to cook here.
                    Man: Oh, I didn't know we couldn't cook in the lounge.
                    Woman: Cooking must be done in the kitchen. You can use it until 9 o'clock. But eating cooked food in the lounge is fine.
                    Man: Oh, really? Thanks for letting me know.
                ### Answer: 
                    1. 여자와 남자는 조리실에 있습니다.
                    2. 남자는 휴게실에서 요리를 했습니다. 
                    3. 휴게실에서 음식을 먹으면 안 됩니다.
                    4. 9시 이후에도 조리실을 이용할 수 있습니다.
                ### Correct Answer: 2
                ### Vietnamese Explain: 
                    1. Người phụ nữ và người đàn ông đang ở phòng nấu ăn.
                    2. Người đàn ông đã nấu ăn trong phòng giải lao.
                    3. Không được ăn uống trong phòng giải lao.
                    4. Sau 9 giờ vẫn có thể sử dụng phòng nấu ăn.
                    --------------------
                    남자: 어, 휴게실에서 요리하면 안 되는지 몰랐어요. (Ồ, tôi không biết là không được nấu ăn trong phòng giải lao.)
                ### English Explain: 
                    1. The woman and the man are in the kitchen.
                    2. The man cooked in the lounge.
                    3. Eating is not allowed in the lounge.
                    4. The kitchen can be used even after 9 o'clock.
                    --------------------
                    <g>남자: 어, 휴게실에서 요리하면 안 되는지 몰랐어요.</g> (Man: Oh, I didn't know we couldn't cook in the lounge.)

            ## Chú ý: 
            - Bạn hãy tạo {number} câu hỏi trắc nghiệm.
            - Sau khi tạo xong, hãy kiểm tra lại xem có câu hỏi nào bị trùng lặp không, nếu có hãy tạo lại câu hỏi đó.
            - Các câu hỏi và câu trả lời chỉ được chứa tiếng Hàn.
            - Hãy đánh số thứ tự bắt đầu từ 91.
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
    with open("output_410010.txt", "w", encoding="utf-8") as f:
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