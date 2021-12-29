import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


class Haptic:
    def __init__(self):
        self.haptic_pin = 18
        self.haptic_freq = 100
        GPIO.setup(self.haptic_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.haptic_pin, self.haptic_freq)
        self.pwm.start(0)

    def vibrate(self, duty_cycle):
        self.pwm.ChangeDutyCycle(duty_cycle)
