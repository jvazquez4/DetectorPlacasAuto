import cv2
import pytesseract
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_and_read_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectar bordes 
    edged = cv2.Canny(blur, 50, 200)

    # Encontrar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        epsilon = 0.018 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Si el contorno tiene 4 vértices, podría ser una placa
            x, y, w, h = cv2.boundingRect(approx)
            plate = gray[y:y + h, x:x + w]

            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(approx.astype("float32"), dst_pts)
            warped = cv2.warpPerspective(gray, M, (w, h))

            # Usar OCR para leer el texto de la placa
            text = pytesseract.image_to_string(warped, config='--psm 8')

            # Filtrar el texto 
            if re.match(r'^[A-Z0-9]{3}-[0-9]{2}-[0-9]{2}$', text.strip()):
                return text.strip(), approx

    return None, None

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text, approx = detect_and_read_plate(frame)

        if text:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, text, (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('License Plate Detection', frame)

        # Salir del bucle con esc
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
