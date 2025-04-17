import os
import numpy
from djitellopy import Tello
from pyapriltags import Detector
import cv2
from simple_pid import PID
from sympy.core import parameters

#Prepare Camera
RESOLUTION = (960, 720)

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

apriltags = Detector(families='tag36h11',
                     nthreads=2,
                     quad_decimate=1.0,
                     quad_sigma=0.0,
                     refine_edges=1,
                     decode_sharpening=0.25,
                     debug=0
                     )
# DONT TOUCH!!


# Helper methods
def areaFromPoints(p1, p2, p3, p4):
    return 1 / 2 * abs(
        (p1[0] * p2[1] - p2[0] * p1[1]) + (p2[0] * p3[1] - p3[0] * p2[1]) + (p3[0] * p4[1] - p4[0] * p3[1]) + (
                p4[0] * p1[1] - p1[0] * p4[1]))


def getPercentageScreenCovered(p1, p2, p3, p4):
    areaTag = areaFromPoints(p1, p2, p3, p4)
    areaFrame = RESOLUTION[0] * RESOLUTION[1]
    return (areaTag / areaFrame) * 100


# End Setup Camera

#Setup Tello
                             # we want the drone to be in the middle
pid = PID(0.3, 0.001, 0.1, setpoint=RESOLUTION[0]/2, output_limits=(-100, 100))
#what apriltag ID should we be looking for

TRACKING_ID = 1

drone = Tello()
drone.connect()
drone.streamon()
drone.set_speed(30)
print("Drone Battery: " + str(drone.get_battery()))
input("ready to fly, use input (APRILTAGS PROGRAM)")
drone.takeoff()
# drone.move_up(100)

while True:
    frame = drone.get_frame_read().frame

    # ret, frame = cam.read()
    # if not ret:
    #     break

    # Detect AprilTags in the image
    tags = apriltags.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), False, None, None)
    foundTag = False
    tagX = 0

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(frame, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0), 3)

        percentCovered = round(getPercentageScreenCovered(tag.corners[0], tag.corners[1], tag.corners[2], tag.corners[3]), 2)
        cv2.putText(frame, "ID:" + str(tag.tag_id) + " | % Covering " + str(
            percentCovered) + "%",
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255) if tag.tag_id == TRACKING_ID else (255, 0, 0),
                    thickness=2)

        # tag centers are floats                                            # r b g
        cv2.circle(frame, center=(int(tag.center[0]), int(tag.center[1])), thickness=8, radius=1, color=(0, 0, 255))

        #draw line from center to point

        cv2.line(frame, (int(RESOLUTION[0]/2), int(RESOLUTION[1]/2)), (int(tag.center[0]), int(RESOLUTION[1]/2)), thickness=4, color=(0, 0, 255))

        if tag.tag_id == TRACKING_ID:
            foundTag = True
            tagX = int(tag.center[0])

    if foundTag:
        print(pid(tagX))
        drone.send_rc_control(0, 10, 0, -int(pid(tagX)))
    else:
        print("")
        drone.send_rc_control(0, -0, 0, 0)



    cv2.imshow('Detected tags', frame)
    cv2.setWindowProperty("Detected tags", cv2.WND_PROP_TOPMOST, 1)

    k = cv2.waitKey(1)
    if k == 27:  # wait for ESC key to exit
        drone.land()
        drone.streamoff()
        cv2.destroyAllWindows()
        break
