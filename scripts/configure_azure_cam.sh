sudo mkdir -p /etc/udev/rules.d
curl -sSL https://github.com/microsoft/Azure-Kinect-Sensor-SDK/raw/develop/scripts/99-k4a.rules \
 | sudo tee /etc/udev/rules.d/99-k4a.rules > /dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger
