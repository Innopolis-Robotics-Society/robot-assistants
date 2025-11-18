# test_installation.py
try:
    import speech_recognition as sr
    print("✓ speech_recognition установлен успешно")
    
    # Проверка доступности микрофонов
    print("Доступные микрофоны:")
    print(sr.Microphone.list_microphone_names())
    
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Попробуйте установить: pip install SpeechRecognition pyaudio")