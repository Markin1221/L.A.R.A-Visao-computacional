from ultralytics import YOLO
import cv2
model = YOLO("yolo26n.pt")
result = model("foto3.jpg", save=True)
print(" [+] concluido" )
