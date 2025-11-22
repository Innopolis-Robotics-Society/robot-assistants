#include <chrono>
#include <memory>
#include <cstdint>

#include <rclcpp/rclcpp.hpp>
#include <ur_msgs/srv/set_io.hpp>
#include "ur_assist/srv/gripper_action.hpp"

using namespace std::chrono_literals;

// Пины, к которым подключён EGP (сейчас: шкафные DOUT0 и DOUT1)
// Если захват сидит на tool-выходах, поменяй на 16 и 17.
static constexpr int8_t GRIPPER_OPEN_PIN  = 0;
static constexpr int8_t GRIPPER_CLOSE_PIN = 1;

class GripperNode : public rclcpp::Node
{
public:
  GripperNode() : Node("gripper_node")
  {
    using GripperAction = ur_assist::srv::GripperAction;
    using SetIO         = ur_msgs::srv::SetIO;

    client_ = this->create_client<SetIO>("/io_and_status_controller/set_io");

    service_ = this->create_service<GripperAction>(
      "gripper_action",
      [this](const std::shared_ptr<GripperAction::Request> req,
             std::shared_ptr<GripperAction::Response> res)
      {
        using SetIOReq = SetIO::Request;

        // Ждём доступности set_io (но без вложенного spin’а)
        if (!client_->wait_for_service(1s)) {
          RCLCPP_ERROR(this->get_logger(), "set_io service not available");
          res->success = false;
          res->message = "set_io not available";
          return;
        }

        auto send_io = [&](int8_t pin, float state) -> bool
        {
          auto sreq = std::make_shared<SetIOReq>();
          sreq->fun   = SetIOReq::FUN_SET_DIGITAL_OUT;
          sreq->pin   = pin;     // 0/1 для шкафа, 16/17 для tool DO
          sreq->state = state;   // 0.0 или 1.0

          // Fire-and-forget: не блокируемся и не создаём второй executor
          client_->async_send_request(sreq);
          return true; // считаем, что запрос ушёл
        };

        bool ok = true;
        if (req->open) {
          // открыть: CLOSE_PIN = 0, OPEN_PIN = 1 (пример)
          ok &= send_io(GRIPPER_CLOSE_PIN, 0.0f);
          ok &= send_io(GRIPPER_OPEN_PIN,  1.0f);
        } else {
          // закрыть
          ok &= send_io(GRIPPER_OPEN_PIN,  0.0f);
          ok &= send_io(GRIPPER_CLOSE_PIN, 1.0f);
        }

        res->success = ok;
        res->message = ok ? "OK" : "Failed to set IO";
      });
  }

private:
  rclcpp::Client<ur_msgs::srv::SetIO>::SharedPtr client_;
  rclcpp::Service<ur_assist::srv::GripperAction>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GripperNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
