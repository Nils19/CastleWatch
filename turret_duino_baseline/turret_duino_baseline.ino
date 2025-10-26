#include <Servo.h>

Servo servoX;
Servo servoY;

int posX = 90;
int posY = 90;

int commandX = 0;
int commandY = 0;

// Add smoothing variables
float smoothedX = 90.0;
float smoothedY = 90.0;
float alpha = 1.0; // Smoothing factor (0.1 = heavy smoothing, 0.9 = light smoothing)

// Add deadband threshold
int deadband = 0.02; // Ignore movements smaller than this

int servoXPin = 9;
int servoYPin = 10;

void setup() {
  Serial.begin(9600);
  servoX.attach(servoXPin);
  servoY.attach(servoYPin);

  // Center servos
  servoX.write(0);
  servoY.write(0);
  delay(1000);
  servoX.write(90);
  servoY.write(90);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      String xStr = data.substring(0, commaIndex);
      String yStr = data.substring(commaIndex + 1);

      commandX = xStr.toFloat();
      commandY = yStr.toFloat();

      // Apply deadband - only move if command is significant
      if (abs(commandX) > deadband || abs(commandY) > deadband) {
        int targetX = posX + commandX;
        int targetY = posY + commandY;

        smoothedX = alpha * targetX + (1.0 - alpha) * smoothedX;
        smoothedY = alpha * targetY + (1.0 - alpha) * smoothedY;

        posX = constrain((int)smoothedX, 0, 180);
        posY = constrain((int)smoothedY, 0, 180);

        servoX.write(posX);
        servoY.write(posY);

        delay(20);
      }
    }
  }
}
