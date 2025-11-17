import re
from difflib import SequenceMatcher
from rclpy.node import Node

class SpeechUtils:
    """Утилиты для обработки речи."""
    
    def __init__(self, node: Node):
        self.node = node
        self.tools_dict = {
            'посмотри': ['посмотри', 'изучи', 'осмотри', 'взглянги'],
            'собачка': ['собака', 'собаку', 'собачку'],
            'назад': ['вернись', 'начало', 'назад']
        }

    @staticmethod
    def similarity(a, b):
        """Вычисляет схожесть двух строк."""
        return SequenceMatcher(None, a, b).ratio()

    def find_tools_in_command(self, command, similarity_threshold=0.7):
        """Поиск инструментов в голосовой команде."""
        command = command.lower().strip()
        found_tools = []
        
        # Быстрый поиск по подстроке
        for tool, keywords in self.tools_dict.items():
            for keyword in keywords:
                if keyword in command:
                    found_tools.append(tool)
                    break
        
        if found_tools:
            return list(set(found_tools))
        
        # Поиск по схожести для случаев с ошибками
        words = re.findall(r'\w+', command)
        for word in words:
            for tool, keywords in self.tools_dict.items():
                for keyword in keywords:
                    if self.similarity(word, keyword) >= similarity_threshold:
                        found_tools.append(tool)
                        break
        
        return list(set(found_tools))
