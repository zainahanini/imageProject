import cv2
import numpy as np

def quantize_gray(img, levels=4):
    step = 256 // levels
    return (img // step) * step

def cartoon_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    return cv2.bitwise_and(color, color, mask=edges)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

mode = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    if mode == 1:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == 2:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        quantized = quantize_gray(gray)
        output = cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)
    elif mode == 3:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        output = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    elif mode == 4:
        output = cv2.GaussianBlur(output, (9, 9), 0)
    elif mode == 5:
        output = cartoon_filter(output)

    cv2.imshow('video', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('4'):
        mode = 4
    elif key == ord('5'):
        mode = 5
    elif key == ord('e') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
