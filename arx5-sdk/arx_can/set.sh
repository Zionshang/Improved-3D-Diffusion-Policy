sudo cp arx_can.rules /etc/udev/rules.d/
sudo chmod +x /etc/udev/rules.d/arx_can.rules
sudo udevadm control --reload-rules && sudo udevadm trigger