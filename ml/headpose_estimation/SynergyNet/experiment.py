import os
import cv2
import numpy as np
from synergy3DMM import SynergyNet
model = SynergyNet()
# file_path = '/Users/divyanshnew/Pictures/Photo Booth Library/Pictures/divyansh_dp.jpeg'
# I = cv2.imread(file_path)
# get landmark [[y, x, z], 68 (points)], mesh [[y, x, z], 53215 (points)], and face pose (Euler angles [yaw, pitch, roll] and translation [y, x, z])
# lmk3d, mesh, pose = model.get_all_outputs(I)

# print(f"Landmarks: {lmk3d}")
# print(f"Pose: {pose}")

# pitch, yaw, roll = model.get_pose(I)


# Load a single image and display
def show_angles(vpath):
    video = cv2.VideoCapture(vpath)
    # video = cv2.VideoCapture(0)


    while(True):
        
        # Capture the video frame
        # by frame
        success, frame = video.read()
        if not success:
            break
        # print(type(frame))
        # frame = cv2.resize(frame, (640, 360))
        original_frame = frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(frame)

        # plt.figure(figsize=(12, 8))
        # plt.imshow(frame)
        # plt.axis('off')
        pitch, yaw, roll = model.get_pose(frame)
        message = None
        if (len(pitch) and len(yaw) and len(roll)):
            if yaw[0] > 26:
                message = "Looking right."
            elif yaw[0] < -10:
                message = "Looking left."
            elif pitch[0] > 26:
                message = "Looking upwards."
            elif pitch[0] < -26:
                message = "Looking downwards"
            if message is None:
                message = "Looking Straight"
            cv2.putText(frame, message, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)
            model.draw_axis(frame, yaw[0], pitch[0], roll[0])
        print(f"Pitch: {pitch}, yaw: {yaw}, roll: {roll}")

        # model.(img=frame, yaw=yaw, pitch=pitch, roll=roll, size=250)

        
        # print(f"Face : {face}")
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(250) & 0xFF
        if key == ord("q"):
            break
    # After the loop release the cap object
    # Destroy all the windows
    cv2.destroyAllWindows()


# base_folder = "/Users/divyanshnew/Documents/vision-service/debug_data/liveness_detection/2023-05-30"
# base_folder = "/Users/divyanshnew/Downloads/2023-05-29"
# for ix, file in enumerate(os.listdir(base_folder)):
#     file_path = os.path.join(base_folder, file)
#     show_angles(vpath=file_path)
# show_angles(vpath="/Users/divyanshnew/Downloads/gif-ea54012c-6ee1-4a0e-a592-a776f16347b2-708118- (1).gif")
show_angles(vpath="/Users/divyanshnew/Downloads/2023-05-29/gif-cd474be6-54da-4d85-8702-24a531959c0c-38645-.gif")