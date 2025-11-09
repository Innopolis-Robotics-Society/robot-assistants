import speech_recognition as sr
import re

class FastToolAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.tools_dict = {
            'молоток': ['молоток', 'молот', 'кувалда'],
            'отвертка': ['отвертка', 'отверточка'],
            'гаечный ключ': ['гаечный ключ', 'ключ'],
            'плоскогубцы': ['плоскогубцы', 'пассатижи'],
            'ножовка': ['ножовка', 'пила'],
            'рулетка': ['рулетка', 'метр'],
            'дрель': ['дрель', 'перфоратор'],
            'стамеска': ['стамеска', 'долото'],
            'уровень': ['уровень', 'ватерпас'],
            'нож': ['нож', 'резак']
        }
    
    def fast_find_tool(self, command):
        """Быстрый поиск с использованием регулярных выражений"""
        command = command.lower()
        found_tools = []
        
        print(f"🔍 Анализируем команду: '{command}'")
        
        for tool_name, keywords in self.tools_dict.items():
            # Создаем шаблон для поиска любого из ключевых слов
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            match = re.search(pattern, command)
            if match:
                found_word = match.group(1)
                print(f"   Найдено совпадение: '{found_word}' → инструмент '{tool_name}'")
                found_tools.append(tool_name)
        
        return found_tools
    
    def record_and_process(self):
        """Основной цикл записи и обработки"""
        print("\n" + "="*50)
        command = self.record_and_recognize()
        
        if command:
            print(f"📋 Полный текст команды: '{command}'")
            
            if any(word in command for word in ['стоп', 'выход', 'хватит']):
                print("🛑 Команда завершения работы обнаружена")
                return "exit", []
            
            tools = self.fast_find_tool(command)
            return "continue", tools
        
        return "continue", []
    
    def record_and_recognize(self):
        """Запись и распознавание аудио"""
        with self.microphone:
            self.recognizer.adjust_for_ambient_noise(self.microphone, duration=1)
            
            try:
                print("🎤 Слушаю... (говорите сейчас)")
                audio = self.recognizer.listen(self.microphone, 5, 5)
                print("✅ Аудио записано, начинаю распознавание...")
                command = self.recognizer.recognize_google(audio, language="ru").lower()
                return command
                
            except sr.UnknownValueError:
                print("❌ Речь не распознана (неразборчивый звук)")
                return ""
            except sr.RequestError as e:
                print(f"❌ Ошибка сервиса распознавания: {e}")
                return ""
            except sr.WaitTimeoutError:
                print("❌ Таймаут записи (ничего не сказано)")
                return ""
            except Exception as e:
                print(f"❌ Неожиданная ошибка: {e}")
                return ""

    def show_available_tools(self):
        """Показать доступные инструменты"""
        print("\n📋 Доступные для распознавания инструменты:")
        for i, (tool_name, keywords) in enumerate(self.tools_dict.items(), 1):
            print(f"   {i:2d}. {tool_name:15} → ключевые слова: {', '.join(keywords)}")

if __name__ == "__main__":
    assistant = FastToolAssistant()
    
    print("🔧 БЫСТРЫЙ ГОЛОСОВОЙ ПОМОЩНИК ДЛЯ ИНСТРУМЕНТОВ")
    print("="*50)
    
    try:
        while True:
            status, tools = assistant.record_and_process()
            
            if status == "exit":
                print("\n👋 Завершение работы...")
                break
            
            if tools:
                print(f"🎯 РЕЗУЛЬТАТ: Найдены инструменты → {', '.join(tools)}")
            else:
                print("💡 РЕЗУЛЬТАТ: Инструменты не найдены в команде")
            
            print("\nГотов к следующей команде...")
                
    except KeyboardInterrupt:
        print("\n\n👋 Программа завершена пользователем")