import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import tempfile
import ffmpeg
import shutil
import torch

class TennisShotLabeler:
    def __init__(self, clips_dir, cropped_dir, annotations_file):
        self.clips_dir = clips_dir
        self.cropped_dir = cropped_dir
        self.annotations_file = annotations_file
        self.current_clip = None
        self.current_frame = None
        self.selected_person = None
        self.tracking_box = None
        self.annotations = self.load_annotations()
        
        self.model = YOLO('yolov8n.pt').to('cpu')
        
        self.shot_types = {
            's': 'Serve',
            'f': 'Forehand',
            'b': 'Backhand',
            'l': 'Forehand Slice',
            'k': 'Backhand Slice',
            'v': 'Forehand Volley',
            'w': 'Backhand Volley',
            'm': 'Smash'
        }
        
    def load_annotations(self):
        if os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {self.annotations_file} contains invalid JSON. Starting with an empty dictionary.")
                return {}
        return {}
    
    def save_annotations(self):
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)
    
    def detect_people(self, frame):
        """Detect people using YOLO"""
        boxes = []
        
        results = self.model(frame, classes=[0])
        
        if results and len(results) > 0:
            for result in results[0].boxes.data:
                if len(result) >= 6:
                    x1, y1, x2, y2, conf, cls = result
                    if conf > 0.3:
                        w = x2 - x1
                        h = y2 - y1
                        
                        padding_w = int(w * 0.2)
                        padding_h = int(h * 0.2)
                        x1 = max(0, int(x1 - padding_w))
                        y1 = max(0, int(y1 - padding_h))
                        w = min(frame.shape[1] - x1, int(w + 2*padding_w))
                        h = min(frame.shape[0] - y1, int(h + 2*padding_h))
                        
                        boxes.append((x1, y1, w, h))
        
        return boxes
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_index = None
            for i in reversed(range(len(self.detected_boxes))):
                x1, y1, w, h = self.detected_boxes[i]
                x2, y2 = x1 + w, y1 + h
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_index = i
                    break
            if selected_index is not None:
                self.selected_person = self.detected_boxes[selected_index]
                self.tracking_box = self.selected_person
                print(f"Player selected! (Box #{selected_index+1})")
                return
    
    def process_clips(self):
        if not os.path.exists(self.cropped_dir):
            os.makedirs(self.cropped_dir)
            
        clips = [f for f in os.listdir(self.clips_dir) if f.endswith('.mp4')]
        unlabeled = [c for c in clips if c not in self.annotations]
        
        if not unlabeled:
            print("No unlabeled clips found!")
            return
        
        print(f"Found {len(unlabeled)} unlabeled clips")
        print("\nControls:")
        for key, shot in self.shot_types.items():
            print(f"{key} - {shot}")
        print("del - Delete clip")
        print("esc - Skip clip")
        
        for clip_name in unlabeled:
            if self.label_clip(clip_name):
                self.save_annotations()
    
    def label_clip(self, clip_name):
        clip_path = os.path.join(self.clips_dir, clip_name)
        cap = cv2.VideoCapture(clip_path)
        
        if not cap.isOpened():
            print(f"Error opening clip: {clip_name}")
            return False
        
        self.selected_person = None
        self.tracking_box = None
        
        print("\nPreviewing clip... Press 'space' to pause, 'esc' to delete, 'enter' to continue")
        paused = False
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            cv2.imshow('Preview', frame)
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            
            if key == ord(' '):
                paused = not paused
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                
                original_clip_path = os.path.join(self.clips_dir, clip_name)
                if os.path.exists(original_clip_path):
                    os.remove(original_clip_path)
                    print(f"\nDeleted clip: {clip_name}")
                return False
            elif key == 13:
                break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            return False
        
        frames_to_try = range(0, 30, 5)
        for frame_idx in frames_to_try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            self.detected_boxes = self.detect_people(frame)
            if len(self.detected_boxes) >= 2:
                break
            elif self.detected_boxes:
                continue
        
        if not self.detected_boxes:
            print("No players detected in clip!")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        cv2.namedWindow('Select Player')
        cv2.setMouseCallback('Select Player', self.mouse_callback)
        
        while self.selected_person is None:
            display_frame = frame.copy()
            for i, box in enumerate(self.detected_boxes):
                x, y, w, h = box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, str(i+1), (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Select Player', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()

        tracker = cv2.TrackerCSRT_create()
        x, y, w, h = self.selected_person
        tracker.init(frame, (x, y, w, h))

        def get_crop_window(frame_shape, box, scale=1.0):
            """
            Create a larger crop window around the player
            scale: multiplier for box size (2.0 = twice the size of player bbox)
            """
            frame_h, frame_w = frame_shape[:2]
            x, y, w, h = box

            center_x = x + w/2
            center_y = y + h/2

            new_w = int(w * scale * 3)
            new_h = int(h * scale)

            new_x = int(center_x - new_w/2)
            new_y = int(center_y - new_h/2)

            new_x = max(0, min(new_x, frame_w - new_w))
            new_y = max(0, min(new_y, frame_h - new_h))
            new_w = min(new_w, frame_w - new_x)
            new_h = min(new_h, frame_h - new_y)

            return (new_x, new_y, new_w, new_h)

        try:
            temp_dir = tempfile.mkdtemp()
            temp_output = os.path.join(temp_dir, f"temp_{clip_name}")

            codecs = [
                ('avc1', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('mp4v', '.mp4')
            ]

            out = None
            for codec, ext in codecs:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    x, y, w, h = get_crop_window(frame.shape, self.selected_person, scale=2.0)
                    cropped = frame[y:y+h, x:x+w]
                    h_out, w_out = cropped.shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    output_file = f"{os.path.splitext(temp_output)[0]}{ext}"
                    out = cv2.VideoWriter(output_file, fourcc, 30, (w_out, h_out))

                    if out is not None and out.isOpened():
                        break
                except Exception as e:
                    print(f"Failed with codec {codec}: {str(e)}")
                    if out is not None:
                        out.release()
                    out = None

            if out is None:
                print("Failed to create video writer with any codec")
                cap.release()
                cv2.destroyAllWindows()
                return False

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    crop_x, crop_y, crop_w, crop_h = get_crop_window(frame.shape, (x, y, w, h), scale=2.0)
                    frame_h, frame_w = frame.shape[:2]
                    crop_x = max(0, min(crop_x, frame_w - 1))
                    crop_y = max(0, min(crop_y, frame_h - 1))
                    crop_w = max(1, min(crop_w, frame_w - crop_x))
                    crop_h = max(1, min(crop_h, frame_h - crop_y))
                    if crop_w <= 0 or crop_h <= 0:
                        print("Invalid crop size, skipping frame.")
                        continue

                    cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                    if cropped.shape[0] != h_out or cropped.shape[1] != w_out:
                        cropped = cv2.resize(cropped, (w_out, h_out))

                    out.write(cropped)
                    cv2.imshow('Cropping', cropped)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    print("Tracking failure detected.")
                    break

            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

            cropped_cap = cv2.VideoCapture(temp_output)
            
            if not cropped_cap.isOpened():
                print("Error opening cropped clip for review")
                shutil.rmtree(temp_dir)
                return False
            
            print("\nSelect shot type:")
            for key, shot in self.shot_types.items():
                print(f"{key} - {shot}")
            print("ESC - Skip/Delete clip")
            
            cv2.namedWindow('Review Shot (Press key to label)', cv2.WINDOW_NORMAL)
            labeled = False
            
            while True:
                ret, frame = cropped_cap.read()
                if not ret:
                    cropped_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                cv2.imshow('Review Shot (Press key to label)', frame)
                key = cv2.waitKey(30) & 0xFF
                
                if chr(key) in self.shot_types:
                    shot_type = self.shot_types[chr(key)]
                    labeled = True
                    break
                elif key == 27:
                    break
            
            cropped_cap.release()
            cv2.destroyAllWindows()
            
            if labeled:
                final_output = os.path.join(self.cropped_dir, f" {clip_name}")
                shutil.copy2(temp_output, final_output)

                self.annotations[clip_name] = shot_type
                print(f"\nLabeled as: {shot_type}")

                original_clip_path = os.path.join(self.clips_dir, clip_name)
                if os.path.exists(original_clip_path):
                    os.remove(original_clip_path)
                    print(f"Deleted original clip: {clip_name}")
            else:
                original_clip_path = os.path.join(self.clips_dir, clip_name)
                if os.path.exists(original_clip_path):
                    os.remove(original_clip_path)
                    print(f"\nDeleted original clip: {clip_name}")
            
        finally:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
        
        return labeled

def main():
    clips_dir = "Data/Clips"
    cropped_dir = "Data/Shots"
    annotations_file = "Data/shot_annotations.json"
    
    labeler = TennisShotLabeler(clips_dir, cropped_dir, annotations_file)
    labeler.process_clips()

if __name__ == "__main__":
    main()