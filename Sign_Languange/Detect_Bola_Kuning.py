import cv2
import numpy as np
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])
cap = cv2.VideoCapture(0)
ball_diameter = 5
distance = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cx = x+w//2
            cy = y+h//2
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, f'({cx}, {cy})', (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            pixel_size = max(w, h)
            estimated_distance = (ball_diameter * distance) / pixel_size
            estimated_distance = round(estimated_distance, 1)
            cv2.putText(frame, f'Jarak : {estimated_distance} cm', (x-20, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Yellow Ball Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()