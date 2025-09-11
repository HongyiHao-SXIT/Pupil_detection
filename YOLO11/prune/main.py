import cv2
import pygame
import os
from threading import Thread
from yolo_utils import PupilDetector

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
image_scale = 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)
cap.set(cv2.CAP_PROP_FPS, 20)

if not cap.isOpened():
    raise RuntimeError("Failed to open the camera")

model_path = os.path.join("YOLOv11", "runs", "train", "train-200epoch-v11n.yaml", "weights", "best.pt")
detector = PupilDetector(model_path)

pygame.init()
screen_width, screen_height = 800, 500
sscreen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Pupil Tracking Animation")

image_folder = "images"
images = [pygame.image.load(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if f.endswith(".png")]
if not images:
    raise RuntimeError("No PNG images found. Please check the images folder.")

current_image_index = 0
clock = pygame.time.Clock()
pupil_x, pupil_y = 0.5, 0.5
smooth_factor = 0.3
smooth_x, smooth_y = pupil_x, pupil_y

def update_pupil_coords():
    global pupil_x, pupil_y
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        coords = detector.detect_pupil_center(frame)
        if coords:
            pupil_x, pupil_y = coords

thread = Thread(target=update_pupil_coords, daemon=True)
thread.start()

running = True
while running:
    sscreen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                current_image_index = (current_image_index + 1) % len(images)
            elif event.key == pygame.K_ESCAPE:
                running = False

    smooth_x = smooth_x + smooth_factor * (pupil_x - smooth_x)
    smooth_y = smooth_y + smooth_factor * (pupil_y - smooth_y)

    img = images[current_image_index]
    orig_w, orig_h = img.get_width(), img.get_height()
    scaled_w = int(screen_width * image_scale)
    scaled_h = int(orig_h * (scaled_w / orig_w))
    img_scaled = pygame.transform.smoothscale(img, (scaled_w, scaled_h))

    center_x = int(smooth_x * screen_width) - scaled_w // 2
    center_y = int(smooth_y * screen_height) - scaled_h // 2

    sscreen.blit(img_scaled, (center_x, center_y))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()