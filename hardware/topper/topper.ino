#include <Servo.h>

Servo myservo;

int sensorPin;
float thresh;
float light;
String b;
int state;

void setup() {
  // Open Serial Port
  Serial.begin(9600);
  Serial.setTimeout(50);
  // Attach Servo
  myservo.attach(9);
  // Define Sensor Pin
  sensorPin = 0;
  // Define cutoff threshold
  thresh = 870;

  myservo.write(30);
}

void loop() {
  // put your main code here, to run repeatedly:
  light = analogRead(sensorPin);
  Serial.println(light);

  if (state == 10) {
    myservo.write(30);
    state = 0;
  } else if (state > 0) {
    state++;
  }
  
  b = Serial.readString();
  b.trim();
  
  if (b == "int") {
    myservo.write(15);
    state = 1;
  }

  delay(50);
}
