import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import pyttsx3


VOICE_PHRASES = {
    "take": {
        "default": "I am bringing the {tool}.",
    },
    "put_back": {
        "default": "I am putting the {tool} back.",
    },
    "stop": {
        "default": "Stopping current action.",
    },
    "help": {
        "default": "You can ask me to take, return tools, or stop.",
    }
}


class VoiceFeedbackNode(Node):

    def __init__(self):
        super().__init__('voice_feedback_node')

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)

        self.subscription = self.create_subscription(
            String,
            '/voice/command',
            self.command_callback,
            10
        )

        self.get_logger().info('Voice feedback node started')

    def command_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON received')
            return

        intent = data.get("intent")
        tool = data.get("tool")
        confidence = data.get("confidence", 0.0)

        if intent not in VOICE_PHRASES:
            self.get_logger().warn(f"Unknown intent: {intent}")
            return

        phrase_template = VOICE_PHRASES[intent]["default"]

        if "{tool}" in phrase_template:
            if tool is None:
                self.get_logger().warn("Tool missing in command")
                return
            phrase = phrase_template.format(tool=tool)
        else:
            phrase = phrase_template

        self.get_logger().info(f"Speaking: {phrase}")
        self.speak(phrase)

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()


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

