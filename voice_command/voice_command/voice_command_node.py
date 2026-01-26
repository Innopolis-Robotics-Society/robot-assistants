import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import threading
import queue
import numpy as np
import sounddevice as sd
import torch



# =========================
# Словарь фраз
# =========================
VOICE_PHRASES = {
    "take": "I am bringing the {tool}.",
    "put_back": "I am putting the {tool} back.",
    "stop": "Stopping current action.",
    "help": "You can ask me to take, return tools, or stop.",
    "greeting": "Hello! I am ready to assist you."
}


# =========================
# Класс Kokoro TTS
# =========================

from kokoro import KPipeline


class KokoroTTS:
    def __init__(self, voice="af_heart"):
        self.voice = voice
        self.pipeline = None
        self.sample_rate = 24000

    def initialize(self, logger):
        logger.info("Loading Kokoro TTS model...")
        self.pipeline = KPipeline(lang_code="a")  # auto language
        # прогрев
        _ = self.synthesize("System ready")
        logger.info("✅ Kokoro TTS ready")
        return True

    def synthesize(self, text: str):
        """
        Kokoro возвращает генератор:
        (phonemes, tokens, audio)
        """
        audio = None
        for _, _, audio_chunk in self.pipeline(text, voice=self.voice):
            audio = audio_chunk

        if audio is None:
            raise RuntimeError("Kokoro produced no audio")

        return audio, self.sample_rate

# =========================
# ROS 2 Node
# =========================
class VoiceFeedbackNode(Node):
    def __init__(self):
        super().__init__('voice_feedback_node')

        # Инициализация TTS
        self.tts = KokoroTTS(voice="af_bella")
        self.audio_queue = queue.Queue()
        self.is_playing = False

        # Подписка на топик
        self.subscription = self.create_subscription(
            String,
            '/voice/command',
            self.command_callback,
            10
        )

        # Таймер для воспроизведения
        self.timer = self.create_timer(0.1, self.play_from_queue)

        # Инициализация TTS в отдельном потоке
        threading.Thread(target=self.init_tts, daemon=True).start()

        self.get_logger().info("Voice feedback node started")

    def init_tts(self):
        try:
            self.tts.initialize(self.get_logger())
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Silero TTS: {e}")

    def command_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON")
            return

        intent = data.get("intent")
        tool = data.get("tool", "")

        if intent not in VOICE_PHRASES:
            self.get_logger().warn(f"Unknown intent: {intent}")
            return

        text = VOICE_PHRASES[intent]
        if "{tool}" in text:
            text = text.format(tool=tool or "the tool")

        self.get_logger().info(f"Speaking: {text}")

        # Асинхронная генерация речи
        threading.Thread(
            target=self.generate_audio,
            args=(text,),
            daemon=True
        ).start()

    def generate_audio(self, text):
        try:
            audio, sr = self.tts.synthesize(text)
            self.audio_queue.put((audio, sr, text))
        except Exception as e:
            self.get_logger().error(f"TTS generation failed: {e}")

    def play_from_queue(self):
        if self.is_playing or self.audio_queue.empty():
            return

        audio, sr, text = self.audio_queue.get()
        self.is_playing = True

        def play():
            try:
                self.get_logger().info(f"🔊 Playing: {text}")
                sd.play(audio, sr)
                sd.wait()
            except Exception as e:
                self.get_logger().error(f"Playback failed: {e}")
            finally:
                self.is_playing = False

        threading.Thread(target=play, daemon=True).start()


# =========================
# main
# =========================
def main(args=None):
    rclpy.init(args=args)
    node = VoiceFeedbackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

