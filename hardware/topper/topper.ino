#include <Servo.h>

Servo myservo;

int sensorPin;
float thresh;
float light;
String b;

void setup() {
  // Open Serial Port
  Serial.begin(9600);
  Serial.setTimeout(100);
  // Attach Servo
  myservo.attach(9);
  // Define Sensor Pin
  sensorPin = 0;
  // Define cutoff threshold
  thresh = 870;

}

void loop() {
  // put your main code here, to run repeatedly:
  light = analogRead(sensorPin);
  Serial.println(light);

  myservo.write(10);

  b = Serial.readString();
  b.trim();
  
  if (b == "int") {
    myservo.write(0);
  }

  delay(200);
}
