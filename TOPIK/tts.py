import os
import re
import hashlib
import asyncio
import edge_tts
from pydub import AudioSegment
from typing import List, Tuple, Optional, Dict

class KoreanTextToSpeech:
    def __init__(self, output_dir = "audio_files", sound_effects_dir = "sound_effects"):
        """
        Khởi tạo class với thư mục đầu ra
        
        Args:
            output_dir: Thư mục lưu file audio, mặc định là "audio_files"
            sound_effects_dir: Thư mục chứa file hiệu ứng âm thanh, mặc định là "sound_effects"
        """
        self.output_dir = output_dir
        self.sound_effects_dir = sound_effects_dir
        
        # Tạo thư mục output nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Tạo thư mục sound_effects nếu chưa tồn tại
        if not os.path.exists(sound_effects_dir):
            os.makedirs(sound_effects_dir)
        
        # Đường dẫn đến các file âm thanh
        self.sound_effects = {
            "phone_ring": os.path.join(sound_effects_dir, "phone_ring.wav"),       # Chuông điện thoại (따르릉)
            "ding_dong_dang": os.path.join(sound_effects_dir, "ding_dong_dang.wav"),  # Âm báo đầu câu (딩동댕)
            "dang_dong_ding": os.path.join(sound_effects_dir, "dang_dong_ding.wav")   # Âm báo cuối câu (댕동딩)
        }
        
        # Định nghĩa giọng đọc
        self.voices = {
            "male": "ko-KR-InJoonNeural",  # Giọng nam
            "female": "ko-KR-SunHiNeural"  # Giọng nữ
        }
        
        # Cấu hình tốc độ và cao độ
        self.speech_rate = "-30%"  # -30%
        self.pitch = "+0Hz"  # 0Hz
    
    def calculate_md5_hash(self, string: str) -> str:
        """
        Tính MD5 hash từ chuỗi
        
        Args:
            string: Chuỗi cần tính MD5
        
        Returns:
            Chuỗi MD5 hash
        """
        md5_hash = hashlib.md5()
        md5_hash.update(string.encode('utf-8'))
        return md5_hash.hexdigest()
    
    def _replace_question_marks(self, text: str) -> str:
        """
        Thay thế dấu "?" bằng "? ?????????????????" và dấu "." bằng dấu "," sau MD5 hash nhằm mục đích lên giọng cuối câu + giàm khoảng nghỉ giữa 2 câu trong 1 đoạn
        
        Args:
            text: Văn bản cần xử lý
            
        Returns:
            Văn bản đã thay thế dấu hỏi chấm
        """
        return text.replace("?", "? ?????????????????").replace(".", ",")
    
    def _detect_sound_effects(self, text: str) -> Dict[str, bool]:
        """
        Phát hiện các hiệu ứng âm thanh trong văn bản
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            Dictionary với tên hiệu ứng và giá trị boolean (có hiệu ứng hay không)
        """
        sound_effects = {
            "phone_ring": "(따르릉)" in text,
            "ding_dong_dang": "(딩동댕)" in text,
            "dang_dong_ding": "(댕동딩)" in text
        }
        
        return sound_effects
    
    def _remove_sound_effect_markers(self, text: str) -> str:
        """
        Loại bỏ tất cả các ký hiệu hiệu ứng âm thanh khỏi văn bản
        
        Args:
            text: Văn bản cần xử lý
            
        Returns:
            Văn bản đã loại bỏ các ký hiệu
        """
        text = text.replace("(따르릉)", "").replace("(딩동댕)", "").replace("(댕동딩)", "")
        return text.strip()
    
    def _detect_dialogue(self, text: str) -> List[Tuple[str, str, Dict[str, bool]]]:
        """
        Phát hiện và tách hội thoại thành các phần, đồng thời phát hiện các hiệu ứng âm thanh
        
        Args:
            text: Văn bản đầu vào có thể chứa hội thoại
            
        Returns:
            Danh sách các tuple (giới tính, văn bản, từ điển hiệu ứng âm thanh)
        """
        # Chuẩn hóa văn bản
        text = text.strip()
        
        # Pattern để phát hiện từng dòng đối thoại riêng biệt
        dialogue_pattern = r'(남자|여자)\s*:\s*([^\n]+(?:\n(?!남자:|여자:)[^\n]+)*)'
        
        matches = re.findall(dialogue_pattern, text, re.DOTALL)
        
        if matches:
            # Trả về list các tuple (giới tính, văn bản, từ điển hiệu ứng âm thanh)
            result = []
            for gender, content in matches:
                gender_type = "male" if gender == "남자" else "female"
                
                # Phát hiện các hiệu ứng âm thanh
                sound_effects = self._detect_sound_effects(content)
                
                # Loại bỏ tất cả các ký hiệu hiệu ứng âm thanh
                clean_content = self._remove_sound_effect_markers(content)
                
                result.append((gender_type, clean_content, sound_effects))
            
            return result
        else:
            # Nếu không phải hội thoại, trả về toàn bộ văn bản với giới tính mặc định là nam
            # Phát hiện các hiệu ứng âm thanh
            sound_effects = self._detect_sound_effects(text)
            
            # Loại bỏ tất cả các ký hiệu hiệu ứng âm thanh
            clean_text = self._remove_sound_effect_markers(text)
            
            return [("male", clean_text, sound_effects)]
    
    async def _generate_audio(self, text: str, voice: str, output_file: str) -> str:
        """
        Tạo file audio từ văn bản sử dụng edge-tts
        
        Args:
            text: Văn bản cần chuyển thành âm thanh
            voice: Giọng đọc
            output_file: Đường dẫn file âm thanh đầu ra
            
        Returns:
            Đường dẫn đến file audio đã tạo
        """
        # Thay thế dấu ? thành ? ????????????????? trước khi tạo audio
        text_with_extended_questions = self._replace_question_marks(text)
        
        communicate = edge_tts.Communicate(text_with_extended_questions, voice, rate=self.speech_rate, pitch=self.pitch)
        await communicate.save(output_file)
        return output_file
    
    def _trim_audio(self, file_path: str, trim_ms: int = 1000) -> str:
        """
        Cắt bỏ 1 giây cuối của file audio
        
        Args:
            file_path: Đường dẫn file audio cần cắt
            trim_ms: Số mili giây cần cắt từ cuối, mặc định là 1000 (1 giây)
            
        Returns:
            Đường dẫn đến file audio đã cắt
        """
        audio = AudioSegment.from_file(file_path)
        
        # Đảm bảo không cắt quá độ dài của audio
        if len(audio) > trim_ms:
            trimmed_audio = audio[:-trim_ms]
        else:
            # Nếu audio ngắn hơn 1 giây, giữ nguyên
            trimmed_audio = audio
            
        # Lưu file audio đã cắt
        trimmed_audio.export(file_path, format="mp3")
        return file_path
    
    def _merge_audio_files(self, file_paths: List[str], output_file: str) -> str:
        """
        Ghép nhiều file audio thành một file
        
        Args:
            file_paths: Danh sách đường dẫn các file audio cần ghép
            output_file: Đường dẫn file audio đầu ra sau khi ghép
            
        Returns:
            Đường dẫn đến file audio đã ghép
        """
        if not file_paths:
            return ""
        
        if len(file_paths) == 1:
            os.rename(file_paths[0], output_file)
            return output_file
        
        # Khởi tạo với file audio đầu tiên
        combined = AudioSegment.from_file(file_paths[0])
        
        # Thêm các file còn lại
        for file_path in file_paths[1:]:
            audio = AudioSegment.from_file(file_path)
            combined += audio
            
        # Xuất file ghép
        combined.export(output_file, format="mp3")
        return output_file
        
    async def _generate_dialogue_audio(self, dialogue_parts: List[Tuple[str, str, Dict[str, bool]]], text_md5: str) -> List[str]:
        """
        Tạo file audio cho từng phần hội thoại, thêm các hiệu ứng âm thanh nếu cần
        
        Args:
            dialogue_parts: Danh sách các phần hội thoại (giới tính, văn bản, từ điển hiệu ứng âm thanh)
            text_md5: MD5 hash của văn bản gốc
            
        Returns:
            Danh sách đường dẫn các file audio tạm thời
        """
        temp_files = []
        
        for i, (gender, content, sound_effects) in enumerate(dialogue_parts):
            part_files = []  # Danh sách các file cho phần này
            
            # 1. Thêm hiệu ứng âm thanh đầu câu nếu có (딩동댕)
            if sound_effects.get("ding_dong_dang", False) and os.path.exists(self.sound_effects["ding_dong_dang"]):
                effect_temp_file = os.path.join(self.output_dir, f"temp_dingdongdang_{text_md5}_{i}.mp3")
                effect_audio = AudioSegment.from_file(self.sound_effects["ding_dong_dang"])
                effect_audio.export(effect_temp_file, format="mp3")
                part_files.append(effect_temp_file)
            
            # 2. Thêm hiệu ứng âm thanh chuông điện thoại nếu có (따르릉)
            if sound_effects.get("phone_ring", False) and os.path.exists(self.sound_effects["phone_ring"]):
                effect_temp_file = os.path.join(self.output_dir, f"temp_ring_{text_md5}_{i}.mp3")
                effect_audio = AudioSegment.from_file(self.sound_effects["phone_ring"])
                effect_audio.export(effect_temp_file, format="mp3")
                part_files.append(effect_temp_file)
            
            # 3. Tạo file audio cho phần văn bản
            voice = self.voices[gender]
            speech_temp_file = os.path.join(self.output_dir, f"temp_speech_{text_md5}_{i}.mp3")
            
            await self._generate_audio(content, voice, speech_temp_file)
            
            # Cắt 1 giây cuối của audio văn bản
            self._trim_audio(speech_temp_file)
            part_files.append(speech_temp_file)
            
            # 4. Thêm hiệu ứng âm thanh cuối câu nếu có (댕동딩)
            if sound_effects.get("dang_dong_ding", False) and os.path.exists(self.sound_effects["dang_dong_ding"]):
                effect_temp_file = os.path.join(self.output_dir, f"temp_dangdongding_{text_md5}_{i}.mp3")
                effect_audio = AudioSegment.from_file(self.sound_effects["dang_dong_ding"])
                effect_audio.export(effect_temp_file, format="mp3")
                part_files.append(effect_temp_file)
            
            # 5. Ghép các phần audio của phần hội thoại này
            if len(part_files) > 1:
                combined_part_file = os.path.join(self.output_dir, f"temp_combined_{text_md5}_{i}.mp3")
                combined = AudioSegment.from_file(part_files[0])
                for j in range(1, len(part_files)):
                    combined += AudioSegment.from_file(part_files[j])
                combined.export(combined_part_file, format="mp3")
                temp_files.append(combined_part_file)
                
                # Xóa các file tạm thời của phần này
                for temp_file in part_files:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # Nếu chỉ có một file, thêm trực tiếp vào danh sách
                temp_files.extend(part_files)
        
        return temp_files
        
    async def convert_file(self, file_path: str) -> str:
        """
        Chuyển đổi file văn bản thành file âm thanh
        
        Args:
            file_path: Đường dẫn đến file văn bản tiếng Hàn
            
        Returns:
            Đường dẫn đến file audio đầu ra
        """
        # Đọc nội dung từ file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tạo MD5 cho toàn bộ văn bản ban đầu (trước khi thay dấu hỏi)
        text_md5 = self.calculate_md5_hash(text)
        final_output_path = os.path.join(self.output_dir, f"{text_md5}.mp3")
        
        # Kiểm tra nếu file đã tồn tại
        if os.path.exists(final_output_path):
            return final_output_path
        
        # Phát hiện và tách hội thoại
        dialogue_parts = self._detect_dialogue(text)
        
        # Tạo file audio cho từng phần, bao gồm cả âm thanh hiệu ứng nếu cần
        temp_files = await self._generate_dialogue_audio(dialogue_parts, text_md5)
        
        # Ghép các file audio lại với nhau
        self._merge_audio_files(temp_files, final_output_path)
        
        # Xóa các file tạm nếu không muốn giữ lại từng câu trong đoạn hội thoại.
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return final_output_path

    def convert_file_sync(self, file_path: str) -> str:
        """
        Phiên bản đồng bộ của phương thức convert_file
        
        Args:
            file_path: Đường dẫn đến file văn bản tiếng Hàn
            
        Returns:
            Đường dẫn đến file audio đầu ra
        """
        return asyncio.run(self.convert_file(file_path))


# Ví dụ sử dụng:
if __name__ == "__main__":
    # Khởi tạo class
    tts = KoreanTextToSpeech()
    
    # Chuyển đổi từ file
    input_file = "text/draft.txt"  # Thay đổi đường dẫn tương ứng
    output_path = tts.convert_file_sync(input_file)
    print(f"File audio được lưu tại: {output_path}")