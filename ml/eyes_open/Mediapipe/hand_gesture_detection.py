import os
import sys
sys.path.append(os.getcwd())
from typing import List, Dict, Tuple

from backend.data_access.data_access import DataAccessImage
from backend.models.asset import Coordinates
from vision_utils import video_sampler, viz_utils, vision_logger, image_utils, functional_utils

import mediapipe as mp
import pandas as pd
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = vision_logger.VisionLogger(__name__)

class HandGestureRecognition:
    def __init__(self, model_path: str, debug: bool = False) -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options, 
                                                           running_mode=mp.tasks.vision.RunningMode.IMAGE,
                                                           num_hands=2,
                                                           min_hand_detection_confidence=0.1,
                                                           min_hand_presence_confidence=0.1,
                                                           min_tracking_confidence=0.1)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.debug = debug

    def convert_frame_to_mp_image(self, frame):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    def inference(self, data_access_frames: List[DataAccessImage]):
        results = []
        for ix, data_layer in enumerate(data_access_frames):
            frame = data_layer.get_rgb_image()
            mp_image = self.convert_frame_to_mp_image(frame)
            result = self.recognizer.recognize(mp_image)
            
            frame_result = {"frame_id": data_layer.frame_id, "gestures": []}
            
            if result.gestures:
                for hand_gestures, hand_landmarks in zip(result.gestures, result.hand_landmarks):
                    top_gesture = hand_gestures[0]
                    frame_result["gestures"].append({
                        "name": top_gesture.category_name,
                        "score": top_gesture.score,
                        "landmarks": hand_landmarks
                    })
            
            results.append(frame_result)
            
            if self.debug:
                self.display_frame(frame, frame_result)
                
                if cv2.waitKey(100) or 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        
        return results
    
    def display_frame(self, frame, frame_result):
        for i, gesture in enumerate(frame_result["gestures"]):
            # Draw the gesture name and confidence
            text = f"{gesture['name']} ({gesture['score']:.2f})"
            # cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            text_color = (0, 0, 0)  # Black color
            outline_color = (0, 0, 0)  # Black outline
            
            # Position the text
            x = 20
            y = 60 + i * 60
            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)

            
            # Draw hand landmarks
            landmarks = gesture["landmarks"]
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Connect landmarks to visualize hand skeleton
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = (int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0]))
                end_point = (int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
    
    def hand_gesture_batch(self, data_layer_batch: List[DataAccessImage], **kwargs):
        results = self.inference(data_access_frames=data_layer_batch)
        return results

    def process_camera_feed(self):
        cap = cv2.VideoCapture(0)  # 0 for default camera
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the frame to RGB (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.convert_frame_to_mp_image(rgb_frame)
            
            result = self.recognizer.recognize(mp_image)
            
            frame_result = {"gestures": []}
            
            if result.gestures:
                for hand_gestures, hand_landmarks in zip(result.gestures, result.hand_landmarks):
                    top_gesture = hand_gestures[0]
                    frame_result["gestures"].append({
                        "name": top_gesture.category_name,
                        "score": top_gesture.score,
                        "landmarks": hand_landmarks
                    })
            
            self.display_frame(frame, frame_result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = '/Users/divyanshnew/Downloads/gesture_recognizer.task'
    
    # Create HandGestureRecognition instance
    hand_gesture = HandGestureRecognition(model_path=model_path, debug=True)
    
    # Process camera feed
    hand_gesture.process_camera_feed()

    # video_path = '/Users/divyanshnew/Pictures/Photo Booth Library/Pictures/hand_gestures.mov'
    # data_access_frames = video_sampler.sample_video(video_path=video_path, frame_rate=10)
    
    # # Set debug to True to display frames
    # hand_gesture = HandGestureRecognition(model_path=model_path, debug=True)
    # results = hand_gesture.hand_gesture_batch(data_layer_batch=data_access_frames)
    
    # # Print results
    # for result in results:
    #     print(f"Frame {result['frame_id']}:")
    #     for gesture in result['gestures']:
    #         print(f"  Gesture: {gesture['name']}, Confidence: {gesture['score']:.2f}")