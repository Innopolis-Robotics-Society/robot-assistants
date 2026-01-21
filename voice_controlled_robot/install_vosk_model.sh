#!/bin/bash

echo "📥 Установка модели Vosk для русского языка..."

# Создание директории для моделей
MODEL_DIR="$HOME/vosk-models"
mkdir -p $MODEL_DIR
cd $MODEL_DIR

# Скачивание и распаковка модели
wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
unzip vosk-model-small-ru-0.22.zip
rm vosk-model-small-ru-0.22.zip

echo "✅ Модель установлена в: $MODEL_DIR/vosk-model-small-ru-0.22"
echo "🎯 Для использования укажите путь в launch файле:"
echo "   model_path: $MODEL_DIR/vosk-model-small-ru-0.22"
