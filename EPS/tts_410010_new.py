import os
import re
import asyncio
import edge_tts
from pydub import AudioSegment

# Các giọng đọc mặc định
DEFAULT_VOICES = {
    "male": "ko-KR-InJoonNeural",     # Giọng nam Hàn Quốc
    "female": "ko-KR-SunHiNeural"     # Giọng nữ Hàn Quốc
}

# Cấu hình debug
DEBUG_MODE = True

# Các pattern cần lọc ra khỏi văn bản
FILTER_PATTERNS = [
    r'https?://\S+',       # URL
    r'www\.\S+',           # www links
    r'<[^>]+>',            # HTML tags
    r'\S+@\S+\.\S+',       # Email
    r'[^\w\s가-힣.:?!,"\']'  # Ký tự đặc biệt không phải tiếng Hàn hoặc dấu câu thông dụng
]

def debug_print(message):
    """In thông báo debug nếu chế độ debug được bật"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def clean_text(text):
    """Làm sạch văn bản, loại bỏ URL và các phần không cần thiết"""
    # Loại bỏ các pattern cần lọc
    for pattern in FILTER_PATTERNS:
        text = re.sub(pattern, ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def trim_audio_end(audio_file, duration_ms=1000):
    """Cắt một khoảng thời gian từ cuối của file âm thanh"""
    try:
        audio = AudioSegment.from_mp3(audio_file)
        
        # Đảm bảo không cắt quá giới hạn của file
        if len(audio) <= duration_ms:
            debug_print(f"File quá ngắn để cắt: {audio_file} (độ dài: {len(audio)}ms)")
            return False
        
        # Cắt phần cuối
        trimmed_audio = audio[:-duration_ms]
        
        # Ghi đè lên file cũ
        trimmed_audio.export(audio_file, format="mp3")
        
        debug_print(f"Đã cắt {duration_ms}ms từ cuối file: {audio_file}")
        return True
    except Exception as e:
        print(f"Lỗi khi cắt file âm thanh: {e}")
        return False

async def text_to_speech(text, voice, output_path, rate="+0%", pitch="+0Hz"):
    """
    Chuyển đổi văn bản thành giọng nói và lưu vào file
    """
    try:
        # Làm sạch văn bản
        cleaned_text = clean_text(text)
        if not cleaned_text:
            debug_print(f"Văn bản rỗng sau khi làm sạch: '{text}'")
            return False
        
        # Đảm bảo rate và pitch có định dạng chính xác
        if not rate.startswith('+') and not rate.startswith('-'):
            rate = "+" + rate
        if not pitch.startswith('+') and not pitch.startswith('-'):
            pitch = "+" + pitch
        
        debug_print(f"Tạo giọng nói với: voice={voice}, rate={rate}, pitch={pitch}")
        
        # Truyền cả rate và pitch vào Communicate
        communicate = edge_tts.Communicate(
            cleaned_text, 
            voice,
            rate=rate,     # Thêm tham số rate
            pitch=pitch    # Thêm tham số pitch
        )
        
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"Lỗi khi chuyển văn bản thành giọng nói: {e}")
        print(f"Văn bản gốc: {text}")
        print(f"Văn bản đã làm sạch: {clean_text(text)}")
        print(f"Giọng nói: {voice}, Rate: {rate}, Pitch: {pitch}")
        return False

def merge_audio_files(input_files, output_file):
    """Ghép các file âm thanh thành một file duy nhất"""
    if not input_files:
        return False
    
    try:
        combined = AudioSegment.empty()
        
        for file in input_files:
            if os.path.exists(file):
                audio = AudioSegment.from_mp3(file)
                combined += audio
                # Thêm khoảng lặng 500ms giữa các câu
                combined += AudioSegment.silent(duration=500)
        
        # Lưu file đã ghép
        combined.export(output_file, format="mp3")
        return True
    except Exception as e:
        print(f"Lỗi khi ghép file âm thanh: {e}")
        return False

def extract_text_audio_sections(file_path):
    """Trích xuất các phần Text Audio từ file định dạng cấu trúc"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Tìm các phần Text Audio
        pattern = r'### Text Audio:(.*?)(?=### Vietnamese:|$)'
        sections = re.findall(pattern, content, re.DOTALL)
        
        # Nếu không tìm thấy, thử các pattern khác
        if not sections:
            patterns = [
                r'Text Audio:(.*?)(?=Vietnamese:|English:|$)',
                r'남자:.+?여자:.+?(?=\n\n|$)',
                r'여자:.+?남자:.+?(?=\n\n|$)'
            ]
            
            for pattern in patterns:
                sections = re.findall(pattern, content, re.DOTALL)
                if sections:
                    debug_print(f"Đã tìm thấy {len(sections)} phần dùng pattern phụ: {pattern}")
                    break
        
        # Lọc và chuẩn hóa các phần
        cleaned_sections = []
        for section in sections:
            # Chỉ lấy các dòng có chứa "남자:" hoặc "여자:"
            dialog_lines = []
            for line in section.strip().split('\n'):
                line = line.strip()
                if "남자:" in line or "여자:" in line:
                    dialog_lines.append(line)
            
            if dialog_lines:
                cleaned_sections.append('\n'.join(dialog_lines))
        
        debug_print(f"Tổng số đoạn hội thoại hợp lệ: {len(cleaned_sections)}")
        if cleaned_sections and DEBUG_MODE:
            debug_print(f"Mẫu đoạn hội thoại đầu tiên: {cleaned_sections[0]}")
        
        return cleaned_sections
    except Exception as e:
        print(f"Lỗi khi trích xuất phần Text Audio: {e}")
        return []

def extract_dialogues(file_path):
    """Trích xuất các đoạn hội thoại từ file văn bản thường"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Tìm các đoạn hội thoại trong dấu ngoặc kép
        dialogues = []
        
        # Pattern 1: Tìm đoạn hội thoại trong dấu ngoặc kép
        quote_dialogues = re.findall(r'"([^"]+)"', content)
        for dialogue in quote_dialogues:
            if "남자:" in dialogue or "여자:" in dialogue:
                dialogues.append(dialogue)
        
        # Pattern 2: Tìm các khối hội thoại bắt đầu với 남자: hoặc 여자:
        blocks = re.findall(r'((?:남자:|여자:).+?(?=\n\n|$))', content, re.DOTALL)
        for block in blocks:
            if block.strip() and block not in dialogues:
                dialogues.append(block.strip())
        
        debug_print(f"Tổng số đoạn hội thoại hợp lệ: {len(dialogues)}")
        if dialogues and DEBUG_MODE:
            debug_print(f"Mẫu đoạn hội thoại đầu tiên: {dialogues[0]}")
        
        return dialogues
    except Exception as e:
        print(f"Lỗi khi trích xuất đoạn hội thoại: {e}")
        return []

def detect_file_format(file_path):
    """Phát hiện định dạng của file đầu vào"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Kiểm tra các pattern
        if "### Text Audio:" in content or "Text Audio:" in content:
            return "structured"
        elif "남자:" in content and "여자:" in content:
            return "dialogue"
        else:
            return "unknown"
    except Exception as e:
        print(f"Lỗi khi phát hiện định dạng file: {e}")
        return "unknown"

async def process_file(input_file, male_rate, male_pitch, female_rate, female_pitch, trim_end_duration=1000):
    """Xử lý file đầu vào dựa trên định dạng"""
    print(f"Đang đọc file: {input_file}")
    
    # Xác định định dạng file
    file_format = detect_file_format(input_file)
    print(f"Đã phát hiện định dạng file: {file_format}")
    
    # Trích xuất nội dung dựa trên định dạng
    if file_format == "structured":
        sections = extract_text_audio_sections(input_file)
        print(f"Đã tìm thấy {len(sections)} phần Text Audio.")
    elif file_format == "dialogue":
        sections = extract_dialogues(input_file)
        print(f"Đã tìm thấy {len(sections)} đoạn hội thoại.")
    else:
        print(f"Không thể xác định định dạng file. Thử phân tích như file hội thoại...")
        sections = extract_dialogues(input_file)
        if not sections:
            print("Không tìm thấy nội dung phù hợp trong file!")
            return
    
    if not sections:
        print("Không tìm thấy nội dung phù hợp trong file!")
        return
    
    # Tạo thư mục output nếu chưa tồn tại
    base_dir = "voice_410009"
    os.makedirs(base_dir, exist_ok=True)
    
    # Xử lý từng phần
    for i, section in enumerate(sections, 1):
        # Tạo thư mục cho câu hỏi
        question_dir = os.path.join(base_dir, f"câu{i}")
        os.makedirs(question_dir, exist_ok=True)
        
        # Tách các câu
        lines = section.strip().split('\n')
        
        # Danh sách các file âm thanh riêng lẻ để ghép sau này
        audio_files = []
        
        # Xử lý từng câu
        for j, line in enumerate(lines, 1):
            # Tách phần người nói và nội dung
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) < 2:
                    debug_print(f"Không thể tách người nói và nội dung: {line}")
                    continue
                
                speaker, text = parts
                text = text.strip()
                
                # Bỏ qua câu rỗng
                if not text:
                    debug_print(f"Bỏ qua câu rỗng: {line}")
                    continue
                
                # Chọn giọng đọc và tham số dựa vào người nói
                if '여자' in speaker:  # Phụ nữ (nữ)
                    voice = DEFAULT_VOICES["female"]
                    rate = female_rate
                    pitch = female_pitch
                    speaker_type = "Nữ"
                elif '남자' in speaker:  # Đàn ông (nam)
                    voice = DEFAULT_VOICES["male"]
                    rate = male_rate
                    pitch = male_pitch
                    speaker_type = "Nam"
                else:
                    # Mặc định sử dụng giọng nữ
                    voice = DEFAULT_VOICES["female"]
                    rate = female_rate
                    pitch = female_pitch
                    speaker_type = "Mặc định (Nữ)"
                
                # Đường dẫn output cho file riêng lẻ
                individual_output_file = os.path.join(question_dir, f"câu{i}.{j}.mp3")
                
                # Chuyển đổi văn bản thành giọng nói
                print(f"Đang xử lý câu {i}.{j}: {text} - Giọng: {speaker_type} - Tốc độ: {rate}")
                
                if await text_to_speech(text, voice, individual_output_file, rate, pitch):
                    print(f"Đã lưu file: {individual_output_file}")
                    
                    # Cắt 1 giây cuối của file âm thanh
                    if trim_end_duration > 0:
                        if trim_audio_end(individual_output_file, trim_end_duration):
                            print(f"Đã cắt {trim_end_duration}ms từ cuối file.")
                    
                    # Thêm file vào danh sách để ghép sau này
                    audio_files.append(individual_output_file)
                else:
                    print(f"Không thể chuyển văn bản thành giọng nói: {text}")
        
        # Ghép các file âm thanh thành một file duy nhất
        if audio_files:
            combined_output_file = os.path.join(question_dir, f"câu{i}.mp3")
            print(f"\nĐang ghép các file âm thanh cho câu hỏi {i}...")
            if merge_audio_files(audio_files, combined_output_file):
                print(f"Đã ghép xong và lưu file: {combined_output_file}")
            else:
                print(f"Không thể ghép file âm thanh cho câu hỏi {i}!")

async def list_available_voices():
    """Liệt kê danh sách giọng đọc có sẵn"""
    try:
        voices = await edge_tts.list_voices()
        print("Danh sách giọng đọc có sẵn tiếng Hàn:")
        for voice in voices:
            if "ko-KR" in voice["ShortName"]:  # Lọc ra các giọng tiếng Hàn
                print(f"{voice['ShortName']} - {voice['Gender']}")
    except Exception as e:
        print(f"Lỗi khi lấy danh sách giọng đọc: {e}")

async def main():
    """Hàm chính"""
    print("Chương trình chuyển văn bản thành giọng nói tự động")
    
    print("\nGiọng đọc mặc định:")
    print(f"Nam: {DEFAULT_VOICES['male']}")
    print(f"Nữ: {DEFAULT_VOICES['female']}")
    
    # Thêm tùy chọn xem danh sách giọng đọc
    show_voices = input("Bạn muốn xem danh sách giọng đọc có sẵn không? (y/n, mặc định: n): ").lower() == 'y'
    
    if show_voices:
        await list_available_voices()
        print("")
    
    # Thông số cho giọng nam
    male_rate = input("Nhập tốc độ nói cho giọng nam (%, ví dụ: +10%, -15%, mặc định: +0%): ") or "+0%"
    if not male_rate.endswith("%"):
        male_rate = male_rate + "%"
    
    male_pitch = input("Nhập độ cao giọng nói cho giọng nam (Hz, ví dụ: +10Hz, -5Hz, mặc định: +0Hz): ") or "+0Hz"
    if not male_pitch.endswith("Hz"):
        male_pitch = male_pitch + "Hz"
    
    # Thông số cho giọng nữ
    female_rate = input("Nhập tốc độ nói cho giọng nữ (%, ví dụ: +10%, -15%, mặc định: +0%): ") or "+0%"
    if not female_rate.endswith("%"):
        female_rate = female_rate + "%"
    
    female_pitch = input("Nhập độ cao giọng nói cho giọng nữ (Hz, ví dụ: +10Hz, -5Hz, mặc định: +0Hz): ") or "+0Hz"
    if not female_pitch.endswith("Hz"):
        female_pitch = female_pitch + "Hz"
    
    # Thời gian cắt từ cuối mỗi file
    trim_end_duration = input("Nhập thời gian cắt từ cuối mỗi file (ms, mặc định: 1000ms): ") or "1000"
    try:
        trim_end_duration = int(trim_end_duration)
    except ValueError:
        print("Giá trị không hợp lệ, sử dụng giá trị mặc định 1000ms")
        trim_end_duration = 1000
    
    # Lấy tên file đầu vào từ người dùng
    input_file = input("\nNhập đường dẫn đến file văn bản (mặc định là 'paste.txt'): ") or "paste.txt"
    
    # Xử lý file đầu vào
    await process_file(input_file, male_rate, male_pitch, female_rate, female_pitch, trim_end_duration)
    
    print("\nQuá trình chuyển đổi hoàn tất!")

if __name__ == "__main__":
    asyncio.run(main())