from vibration import Haptic
import time

if __name__ == '__main__':
    motor = Haptic()
    while True:
        motor.vibrate(100)
        time.sleep(1)
        motor.vibrate(50)
        time.sleep(1)
        motor.vibrate(0)
        time.sleep(1)

        motor.vibrate(0)
        time.sleep(1)
        motor.vibrate(50)
        time.sleep(1)
        motor.vibrate(100)
        time.sleep(1)
