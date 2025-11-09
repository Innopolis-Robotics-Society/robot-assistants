import sounddevice as sd
import vosk
import json
import queue
import sys
import re
from difflib import SequenceMatcher

# Словарь инструментов (можно вынести в отдельный файл)
tools_dict = {
    'молоток': ['молоток', 'молот', 'кувалда'],
    'отвертка': ['отвертка', 'отверточка'],
    'гаечный ключ': ['гаечный ключ', 'ключ'],
    'плоскогубцы': ['плоскогубцы', 'пассатижи'],
    'ножовка': ['ножовка', 'пила'],
    'рулетка': ['рулетка', 'метр'],
    'дрель': ['дрель', 'шуруповерт'],
    'стамеска': ['стамеска'],
    'уровень': ['уровень'],
    'нож': ['нож', 'резак']
}

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_tools_in_command(command, similarity_threshold=0.7):
    """Поиск инструментов в команде"""
    command = command.lower().strip()
    found_tools = []
    
    # Быстрый поиск по подстроке
    for tool, keywords in tools_dict.items():
        for keyword in keywords:
            if keyword in command:
                found_tools.append(tool)
                break
    
    if found_tools:
        return list(set(found_tools))
    
    # Поиск по схожести для случаев с ошибками
    words = re.findall(r'\w+', command)
    for word in words:
        for tool, keywords in tools_dict.items():
            for keyword in keywords:
                if similarity(word, keyword) >= similarity_threshold:
                    found_tools.append(tool)
                    break
    
    return list(set(found_tools))

# Голосовой ассистент
device_m = 4
samplerate = 16000
q = queue.Queue()
model = vosk.Model("model_stt/vosk-model-small-ru-0.22")

def q_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def voice_assistant():
    try:
        with sd.RawInputStream(callback=q_callback, 
                             channels=1, 
                             samplerate=samplerate, 
                             device=device_m, 
                             dtype='int16'):
            
            print("🎤 Голосовой ассистент запущен. Говорите...")
            
            rec = vosk.KaldiRecognizer(model, samplerate)
            
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        print(f"\n🗣️ Команда: {text}")
                        
                        # Ищем инструменты в команде
                        tools = find_tools_in_command(text)
                        
                        if tools:
                            print(f"🎯 Распознаны инструменты: {', '.join(tools)}")
                            # Здесь можно добавить логику выполнения действий
                        else:
                            print("❌ Инструменты не распознаны")
                            
                else:
                    # Вывод частичных результатов (опционально)
                    partial_result = json.loads(rec.PartialResult())
                    partial_text = partial_result.get("partial", "").strip()
                    if partial_text:
                        print(f"▌ Слушаю: {partial_text}", end='\r')
                        
    except KeyboardInterrupt:
        print("\n✅ Ассистент остановлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    voice_assistant()