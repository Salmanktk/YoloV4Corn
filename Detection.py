import numpy as np
import cv2
import time
from tracker import EuclideanDistTracker

tracker = EuclideanDistTracker()

#Load Yolo
net = cv2.dnn.readNet("corn-yolov4-tiny.weights", "yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1]for i in net.getUnconnectedOutLayers()]

#Loading image
cap = cv2.VideoCapture("corn.mp4")
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    he, wi, _ = frame.shape

    roi = frame[50:280, 70:360]
    height, width, channel = roi.shape



    #Detecting Objects
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (320, 320), (0,0,0),True, crop = False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #Showing info on screen
    class_ids = []
    confidences = []
    boxes= []


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id=np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.5)
    detections = []
    for i in range (len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str("corn")
            confidence = confidences[i]
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255,255,0),2)

            detections.append([x,y,w,h])

    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x,y,w,h,id = box_id
        cv2.putText(roi, "corn" + " " + str(id), (x, y), font, 2, (255, 255, 0), 2)

    elapsed_time = time.time() - starting_time
    fps = (frame_id/elapsed_time)
    cv2.putText(frame, "FPS " + str(round(fps,2)), (10, 50), font, 2, (0,0,0), 2)
    cv2.putText(frame, "COUNT = " + str(id), (10, 80), font, 2, (0,0,0), 2)
    cv2.imshow("image", frame)
    cv2.imshow("roi",roi)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

