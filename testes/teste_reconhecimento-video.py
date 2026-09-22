from ultralytics import YOLO
model = YOLO("yolo26n.pt")

results = model.predict(source="video.mp4", save=True, device=0)
print('[+] deu certo pae')