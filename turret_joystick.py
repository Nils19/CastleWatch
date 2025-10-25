from __future__ import annotations

import time

import pygame
import serial


def scale_axis(value, deadzone=0.65, scaling=2):
    if abs(value) < deadzone:
        return 0.0
    return value * scaling


joystick_threshold = 0.9

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print('No joystick detected!')
    pygame.quit()
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Detected joystick: {joystick.get_name()}")

# Adjust this to your Arduino COM port
arduino_port = 'COM8'  # Windows example
baud_rate = 9600

# Open serial connection
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

try:
    while True:
        pygame.event.pump()

        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        buttons = [
            joystick.get_button(i) for i in range(joystick.get_numbuttons())
        ]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        print(axes)
        print(buttons)
        print(hats)
        print('-' * 50)
        time.sleep(0.1)

        xPos = axes[0]
        yPos = axes[1]

        command_x = -(scale_axis(xPos))
        command_y = -(scale_axis(yPos))

        ser.write(f"{command_x:.2f},{command_y:.2f}\n".encode())

except KeyboardInterrupt:
    print('Exiting...')

finally:
    pygame.quit()
