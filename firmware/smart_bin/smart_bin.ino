// EcoConnect-X - smart community bin
// ESP32 + HX711 + load cell. Posts the bin weight every 10 minutes to /api/ingest/bin.
#include <WiFi.h>
#include <HTTPClient.h>
#include "HX711.h"

const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASS = "YOUR_PASS";
const char* SERVER    = "http://YOUR_SERVER:5000/api/ingest/bin";
const char* BIN_ID    = "B000";
const float CALIBRATION = -7050.0;  // set with a known weight

HX711 scale;

void setup() {
  Serial.begin(115200);
  scale.begin(4, 5);  // DOUT, SCK
  scale.set_scale(CALIBRATION);
  scale.tare();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  float kg = scale.get_units(10);
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER);
    http.addHeader("Content-Type", "application/json");
    http.POST("{\"bin_id\":\"" + String(BIN_ID) + "\",\"weight_kg\":" + String(kg, 2) + "}");
    http.end();
  }
  esp_sleep_enable_timer_wakeup(10ULL * 60 * 1000000);
  esp_deep_sleep_start();
}
