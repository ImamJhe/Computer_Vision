import cv2
import torch
import numpy as np
from tracker import *
from warnings import filterwarnings
filterwarnings("ignore")


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap=cv2.VideoCapture(r"D:\Imam\Belajar\Python\Computer_Vision\Counting_Vehicles\highway.mp4")

count = 0
tracker = Tracker()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        # print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
area_1 = set()
area_2 = set()
count_car = set()

area1 = [(285, 450), (250, 470), (545, 490), (545, 460)]
area2 = [(585, 460), (590, 490), (930, 480), (880, 455)]

# Initialize video writer for the processed video
result_video_path = r'D:\Imam\Belajar\Python\Computer_Vision\Counting_Vehicles\result.mp4'  # Modify the path and filename as needed
fps = 30  # Modify if the frame rate is different
w, h = 1020, 600  # Modify with the desired width and height
result_video_writer = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))
    results = model(frame)
    list = []

    for index, row in results.pandas().xyxy[0].iterrows():
        x = int(row[0])
        y = int(row[1])
        x1 = int(row[2])
        y1 = int(row[3])
        b = str(row['name'])
        if b not in ["person", "motorcycle"]:
            list.append([x, y, x1, y1])
    idx_bbox = tracker.update(list)
    for bbox in idx_bbox:
        x2, y2, x3, y3, id = bbox
        # cv2.rectangle(frame, (x2, y2), (x3, y3), (0, 0, 255), 2)
        cv2.circle(frame, (x3, y3), 4, (0, 255, 0), -1)
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((x3, y3)), False)
        result1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x3, y3)), False)
        # cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 1)
        if result > 0:
            area_1.add(id)
            count_car.add(id)
        if result1 > 0:
            area_2.add(id)
            count_car.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)
    a = len(area_1)
    b = len(area_2)
    count_car1 = len(count_car)
    cv2.putText(frame, str(a), (220, 450), cv2.FONT_HERSHEY_PLAIN, 3, (210, 250, 250), 2)
    cv2.putText(frame, str(b), (925, 465), cv2.FONT_HERSHEY_PLAIN, 3, (210, 250, 250), 2)
    cv2.putText(frame, f'Count = {str(count_car1)}', (50, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    result_video_writer.write(frame)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

result_video_writer.release()
cap.release()
cv2.destroyAllWindows()
