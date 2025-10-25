#include <Servo.h>

Servo servoX;
Servo servoY;

int posX = 90;
int posY = 90;

int commandX = 0;
int commandY = 0;

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
    String data = Serial.readStringUntil('\n'); // Read until newline
    data.trim();

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      String xStr = data.substring(0, commaIndex);
      String yStr = data.substring(commaIndex + 1);

      commandX = xStr.toInt(); // Convert to integer
      commandY = yStr.toInt();

      // Update servo positions based on error
      posX += int(commandX);
      posY += int(commandY);

      // Constrain to valid servo range
      posX = constrain(posX, 0, 180);
      posY = constrain(posY, 0, 180);

      // Move servos
      servoX.write(posX);
      servoY.write(posY);
    }
  }
}
