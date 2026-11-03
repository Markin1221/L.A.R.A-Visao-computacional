// ESP32-S3 CAM - mostra a imagem da camera no navegador
// Tools: Board "ESP32S3 Dev Module" | PSRAM "OPI PSRAM" | USB CDC On Boot "Enabled" | Core Debug Level "Info"

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ===== CONFIGURE AQUI =====
//const char* WIFI_NOME  = "TIM ULTRAFIBRA_E570";
//const char* WIFI_SENHA = "KKrt@71tkT";
const char* WIFI_NOME  = "Markin"
const char* WIFI_SENHA = "markin1221";
const bool  VIRAR_IMAGEM = false;              // true se a imagem aparecer de cabeca pra baixo
// ==========================

// pinos da ESP32-S3-EYE / Freenove ESP32-S3 CAM (maioria das S3-CAM)
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y9_GPIO_NUM    16
#define Y8_GPIO_NUM    17
#define Y7_GPIO_NUM    18
#define Y6_GPIO_NUM    12
#define Y5_GPIO_NUM    10
#define Y4_GPIO_NUM     8
#define Y3_GPIO_NUM     9
#define Y2_GPIO_NUM    11
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

#define BOUNDARY "frame"
static const char* STREAM_TYPE = "multipart/x-mixed-replace;boundary=" BOUNDARY;
static const char* STREAM_SEP  = "\r\n--" BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t servidor = NULL;

// pagina inicial: so mostra o stream
static esp_err_t paginaHandler(httpd_req_t* req) {
  static const char html[] = R"(<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ESP32-S3 CAM</title>
<style>body{margin:0;min-height:100vh;background:#0b0d12;color:#e6e8ee;font-family:sans-serif;
display:flex;flex-direction:column;align-items:center;gap:12px;padding:20px;box-sizing:border-box}
img{max-width:100%;border-radius:12px;border:1px solid #222734}</style></head>
<body><h3>L.A.R.A &middot; ESP32-S3 CAM</h3><img src="/stream"></body></html>)";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

// stream MJPEG: manda um JPEG atras do outro
static esp_err_t streamHandler(httpd_req_t* req) {
  char cabecalho[64];
  esp_err_t res = httpd_resp_set_type(req, STREAM_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      log_e("falha ao capturar frame");
      res = ESP_FAIL;
      break;
    }
    size_t n = snprintf(cabecalho, sizeof(cabecalho), STREAM_PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, STREAM_SEP, strlen(STREAM_SEP));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, cabecalho, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;   // navegador fechou
  }
  return res;
}

bool iniciarCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;      // 640x480
  c.jpeg_quality = 12;               // 10 = melhor, 30 = pior
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;

  if (!psramFound()) {
    log_e("AVISO: PSRAM nao encontrada. Ative Tools > PSRAM > OPI PSRAM.");
    c.frame_size = FRAMESIZE_QVGA;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_DRAM;
    c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    log_e("ERRO ao iniciar a camera: 0x%x (confira o cabo flat e o modelo da placa)", err);
    return false;
  }

  if (VIRAR_IMAGEM) {
    sensor_t* s = esp_camera_sensor_get();
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (!iniciarCamera()) return;
  log_e("camera ok");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_NOME, WIFI_SENHA);
  unsigned long inicio = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - inicio > 20000) {
      log_e("ERRO: nao conectou no Wi-Fi. Confira nome/senha (rede 2.4 GHz).");
      return;
    }
    delay(500);
  }

  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 80;
  if (httpd_start(&servidor, &cfg) == ESP_OK) {
    httpd_uri_t pagina = {};
    pagina.uri = "/";
    pagina.method = HTTP_GET;
    pagina.handler = paginaHandler;
    httpd_register_uri_handler(servidor, &pagina);

    httpd_uri_t stream = {};
    stream.uri = "/stream";
    stream.method = HTTP_GET;
    stream.handler = streamHandler;
    httpd_register_uri_handler(servidor, &stream);
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    log_e("CAMERA EM: http://%s", WiFi.localIP().toString().c_str());
  } else {
    log_e("Wi-Fi NAO conectado");
  }
  delay(5000);
}