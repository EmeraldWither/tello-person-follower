import cv2
import djitellopy
from djitellopy import Tello
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

model = YOLO('yolov8n.pt')

# init tello
tello: Tello = djitellopy.Tello()
tello.connect()
tello.streamon()

tello.set_speed(50)
print("Battery is " + str(tello.get_battery()))
input("ready to fly, use input")

tello.takeoff()
tello.move_up(100)

while True:
    cv2.waitKey(1)
    frame = tello.get_frame_read().frame
    # frame = cv2.resize(frame, (150, 150))

    results: list[Results] = model(frame, )
    # filter results to see if they are a person
    annotated_frame = frame
    boxes = results[0].boxes  # Boxes object for bbox outputs
    # print(boxes)
    for box in boxes:
        if model.names.get(box.cls.item()) == 'person':
            # print(str(box.xyxy))
            x1, y1, x2, y2 = box.xyxy.numpy()[0][0].item(), box.xyxy.numpy()[0][1].item(), box.xyxy.numpy()[0][
                2].item(), box.xyxy.numpy()[0][3].item()
            # print(x1, y1, x2, y2)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            center = int(x1 + (abs(x1 - x2) / 2))
            # get distance from center
            cv2.circle(annotated_frame, (center, int(y1 + (abs(y1 - y2) / 2))), 15, (255, 0, 0), 5)

            # perform tello movements
            if center < 170:
                tello.send_rc_control(0, 0, 0, -30)
            elif center > 450:
                tello.send_rc_control(0, 0, 0, 30)
            else:
                tello.send_rc_control(0, 40, 0, 0)
            break

        # we dont see anything

        # do a scan
        tello.send_rc_control(0, 0, 0, 10)

    # Display thte annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
exit()
