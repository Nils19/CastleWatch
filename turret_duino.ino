#include <Servo.h>

Servo servoX;
Servo servoY;

float posX = 90.0;  // Use float for smoother interpolation
float posY = 90.0;
float targetX = 90.0;
float targetY = 90.0;

int servoXPin = 9;
int servoYPin = 10;

// Smoothing factor (0.1 = very smooth, 0.9 = very responsive)
float smoothing = 0.15;

// Minimum movement threshold to reduce jitter
float deadzone = 0.5;

void setup() {
  Serial.begin(9600);
  servoX.attach(servoXPin);
  servoY.attach(servoYPin);

  // Center servos
  servoX.write(90);
  servoY.write(90);
  delay(1000);
  
  Serial.println("Arduino Ready - Servos Centered");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    
    Serial.print("Received: ");
    Serial.println(data);

    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      String xStr = data.substring(0, commaIndex);
      String yStr = data.substring(commaIndex + 1);

      float commandX = xStr.toFloat();
      float commandY = yStr.toFloat();
      
      Serial.print("Commands - X: ");
      Serial.print(commandX);
      Serial.print(" Y: ");
      Serial.println(commandY);

      // Update target positions
      targetX += commandX;
      targetY += commandY;

      // Constrain targets
      targetX = constrain(targetX, 0, 180);
      targetY = constrain(targetY, 0, 180);
      
      Serial.print("New Targets - X: ");
      Serial.print(targetX);
      Serial.print(" Y: ");
      Serial.println(targetY);
    }
  }

  // Smooth movement towards target
  float deltaX = targetX - posX;
  float deltaY = targetY - posY;
  
  Serial.print("Deltas - X: ");
  Serial.print(deltaX);
  Serial.print(" Y: ");
  Serial.println(deltaY);

  // Apply deadzone to reduce micro-movements
  if (abs(deltaX) > deadzone) {
    posX += deltaX * smoothing;
    Serial.print("Moving X from ");
    Serial.print(posX - deltaX * smoothing);
    Serial.print(" to ");
    Serial.println(posX);
  } else {
    Serial.println("X in deadzone");
  }
  
  if (abs(deltaY) > deadzone) {
    posY += deltaY * smoothing;
    Serial.print("Moving Y from ");
    Serial.print(posY - deltaY * smoothing);
    Serial.print(" to ");
    Serial.println(posY);
  } else {
    Serial.println("Y in deadzone");
  }

  // Constrain current positions
  posX = constrain(posX, 0, 180);
  posY = constrain(posY, 0, 180);

  // Write to servos
  servoX.write((int)posX);
  servoY.write((int)posY);
  
  Serial.print("Final Positions - X: ");
  Serial.print((int)posX);
  Serial.print(" Y: ");
  Serial.println((int)posY);
  Serial.println("---");

  // Small delay for smooth operation
  delay(20);  // 50Hz update rate
}