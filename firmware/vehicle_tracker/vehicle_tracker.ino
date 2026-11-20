// EcoConnect-X - collection-vehicle GPS tracker
// ESP32 + NEO-6M GPS (UART2, RX=16, TX=17). Posts a ping every 15 s to /api/ingest/gps.
// Pings are buffered in RAM when WiFi/4G hotspot drops and flushed when it returns.
#include <WiFi.h>
#include <HTTPClient.h>
#include <TinyGPSPlus.h>

const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASS";
const char* SERVER    = "http://YOUR_SERVER:5000/api/ingest/gps";
const char* VEHICLE   = "KA19-T1";

TinyGPSPlus gps;
HardwareSerial GPSSerial(2);
String buffer = "";
int buffered = 0;
unsigned long lastPing = 0;

void setup() {
  Serial.begin(115200);
  GPSSerial.begin(9600, SERIAL_8N1, 16, 17);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  while (GPSSerial.available()) gps.encode(GPSSerial.read());
  if (millis() - lastPing < 15000 || !gps.location.isValid()) return;
  lastPing = millis();

  String ping = "{\"vehicle_id\":\"" + String(VEHICLE) + "\",\"lat\":" + String(gps.location.lat(), 6) +
                ",\"lon\":" + String(gps.location.lng(), 6) + ",\"speed_kmph\":" + String(gps.speed.kmph(), 1) + "}";
  if (buffered < 200) { buffer += (buffered ? "," : "") + ping; buffered++; }

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER);
    http.addHeader("Content-Type", "application/json");
    if (http.POST("[" + buffer + "]") == 200) { buffer = ""; buffered = 0; }
    http.end();
  }
}
