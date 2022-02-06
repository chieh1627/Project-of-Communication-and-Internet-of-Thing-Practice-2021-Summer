#include <SPI.h>
#include <MFRC522.h> 
#define RESET   9  
#define SS      10

MFRC522 mfrc522(SS, RESET); 
void setup() {
  Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();
}

void loop() {
  if (mfrc522.PICC_IsNewCardPresent()) {
    if (mfrc522.PICC_ReadCardSerial()){
      byte *id = mfrc522.uid.uidByte;
      byte idSize = mfrc522.uid.size; 
      for (byte i = 0; i < idSize; i++) {
        Serial.print(id[i], HEX); 
      }
      Serial.println();
    }
    mfrc522.PICC_HaltA(); 
  } 
}