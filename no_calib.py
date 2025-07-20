#!/usr/bin/python3.10

import os
from l2cs.gaze_detectors import Gaze_Detector
import torch
from copy import deepcopy
import math
import pdb
import cv2
import numpy as np
from time import time, sleep, strftime, localtime
from collections import deque
from threading import Thread, Lock

import yaml

class AttentionDetector:
    def __init__(self, 
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 pitch_threshold=25,       # Increased pitch angle for attention (was 15)
                 yaw_threshold=30,         # Increased yaw angle for attention (was 20)
                 vertical_allowance=10,    # Additional vertical tolerance
                 history_size=10):         # Number of frames to keep for smoothing
        
        # Initialize the gaze detector
        self.gaze_detector = Gaze_Detector(
            device='cuda',
            nn_arch='ResNet50',
            weights_pth='/home/vscode/gaze_ws/L2CSNet_gaze360.pkl'
        )

        # Initialize parameters with more permissive thresholds
        self.attention_threshold = attention_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.vertical_allowance = vertical_allowance
        self.attention_start_time = None
        self.attention_state = False
        
        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)
        
    
    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """
        Determine if the person is looking at the robot based on angles
        Uses asymmetric thresholds with extra vertical allowance
        """
        # Allow more downward looking (negative pitch) than upward
        pitch_upper_limit = self.pitch_threshold
        pitch_lower_limit = self.pitch_threshold + self.vertical_allowance
        
        pitch_ok = -pitch_lower_limit <= pitch <= pitch_upper_limit
        yaw_ok = abs(yaw) < self.yaw_threshold
        
        return pitch_ok and yaw_ok
    
    def process_frame(self, frame):
        # Initialize return values
        attention_detected = False
        sustained_attention = False
        angles = None
        face_found = False

        """Process a single frame and return attention state and visualization"""
        h, w, _ = frame.shape
        
        g_success = self.gaze_detector.detect_gaze(frame)

        if g_success:
            face_found = True
            results = self.gaze_detector.get_latest_gaze_results()
            if results is None:
                return frame, attention_detected, sustained_attention, angles, face_found

            # Extract pitch and yaw from results
            pitch = results.pitch[0]
            yaw = results.yaw[0]

            angles = (math.degrees(pitch), math.degrees(yaw), 0) # Assuming roll is not used and angles are in degrees
            
            # Apply smoothing
            smoothed_angles = self.smooth_angles(angles)
            pitch, yaw, _ = smoothed_angles
            
            # Check if looking at robot
            attention_detected = self.is_looking_at_robot(pitch, yaw)
            
            # Track sustained attention
            current_time = time()
            if attention_detected:
                if self.attention_start_time is None:
                    self.attention_start_time = current_time
                elif (current_time - self.attention_start_time) >= self.attention_threshold:
                    sustained_attention = True
            else:
                self.attention_start_time = None

            frame = self.gaze_detector.draw_gaze_window()

            # Visualization
            color = (0, 255, 0) if sustained_attention else (
                (0, 165, 255) if attention_detected else (0, 0, 255)
            )
            
            # Add text overlays
            cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw attention status
            status = "Sustained Attention" if sustained_attention else (
                "Attention Detected" if attention_detected else "No Attention"
            )
            cv2.putText(frame, status, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display threshold information
            cv2.putText(frame, f'Pitch Range: [-{self.pitch_threshold + self.vertical_allowance}, {self.pitch_threshold}]', 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Yaw Range: [-{self.yaw_threshold}, {self.yaw_threshold}]', 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return frame, attention_detected, sustained_attention, angles, face_found

def calculate_attention_metrics(attention_window, interval_duration=5.0):
    """
    Calculate attention metrics for a given time window of attention data.
    
    Args:
        attention_window (list): List of tuples (timestamp, attention_state)
        interval_duration (float): Duration of analysis interval in seconds
        
    Returns:
        dict: Dictionary containing attention metrics:
            - attention_ratio: Ratio of frames with attention detected
            - gaze_entropy: Shannon entropy of gaze distribution
            - frames_in_interval: Number of frames in analyzed interval
            - robot_looks: Number of frames looking at robot
            - non_robot_looks: Number of frames not looking at robot
    """
    if not attention_window:
        return {
            'attention_ratio': 0.0,
            'gaze_entropy': 0.0,
            'frames_in_interval': 0,
            'robot_looks': 0,
            'non_robot_looks': 0,
            'gaze_score': 0.0
        }
    
    # Get current time and filter window to only include last interval_duration seconds
    current_time = attention_window[-1][0]  # Latest timestamp
    filtered_window = [(t, a) for t, a in attention_window 
                      if current_time - t <= interval_duration]
    
    # Count frames
    frames_in_interval = len(filtered_window)
    robot_looks = sum(1 for _, attention in filtered_window if attention)
    non_robot_looks = frames_in_interval - robot_looks
    
    # Calculate attention ratio
    attention_ratio = robot_looks / frames_in_interval if frames_in_interval > 0 else 0.0
    
    # Calculate stationary gaze entropy
    gaze_entropy = 0.0
    if frames_in_interval > 0:
        p_robot = robot_looks / frames_in_interval
        p_non_robot = non_robot_looks / frames_in_interval
        
        # Calculate entropy using Shannon formula
        if p_robot > 0:
            gaze_entropy -= p_robot * math.log2(p_robot)
        if p_non_robot > 0:
            gaze_entropy -= p_non_robot * math.log2(p_non_robot)
    
    # Compute gaze score using the formula
    normalized_attention_ratio = min(attention_ratio, 1.0)
    normalized_entropy = 1.0 - min(gaze_entropy, 1.0)

    if gaze_entropy == 1.0 or (robot_looks > 30 and 1.0 > gaze_entropy > 0.7):
        gaze_score = 100 * normalized_attention_ratio
    else:
        gaze_score = 100 * (normalized_attention_ratio * normalized_entropy)
    
    gaze_score = max(0, min(100, gaze_score))  # Ensure score is within 0-100
    
    return {
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks,
        'gaze_score': gaze_score
    }

class SimpleGazeController:
    def __init__(self, camera_id=2, 
                 pitch_threshold=25, 
                 yaw_threshold=30, 
                 vertical_allowance=10):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)
        self.detector = AttentionDetector(
            pitch_threshold=pitch_threshold,
            yaw_threshold=yaw_threshold,
            vertical_allowance=vertical_allowance
        )
        self.is_in_attention_detection_mode = False
        self.attention_window_lock = Lock()
        self.gaze_score = 0.0
        self.gaze_score_lock = Lock()
        self.attention_thread = Thread(target=self.attention_detection_loop)
        self.robot_looks_lock = Lock()
        self.robot_looks = 0
        self.gaze_entropy_lock = Lock()
        self.gaze_entropy = 0.0
        self.visualisation_frame = None
        
    def get_gaze_score(self):
        with self.gaze_score_lock:
            return self.gaze_score
    
    def get_robot_looks(self):
        with self.robot_looks_lock:
            return self.robot_looks
    
    def get_gaze_entropy(self):
        with self.gaze_entropy_lock:
            return self.gaze_entropy
    
    def get_visualisation_frame(self):
        with self.gaze_score_lock:
            if self.visualisation_frame is not None:
                return self.visualisation_frame.copy()
            return None
    
    def kill_attention_thread(self):
        self.is_in_attention_detection_mode = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        if self.attention_thread.is_alive():
            self.attention_thread.join()
        
    def start_detecting_attention(self):
        print("Starting attention detection without calibration...")
        print(f"Using thresholds - Pitch: ±{self.detector.pitch_threshold}° (with {self.detector.vertical_allowance}° downward allowance), Yaw: ±{self.detector.yaw_threshold}°")
        self.is_in_attention_detection_mode = True
        self.attention_thread.start()
        
    def attention_detection_loop(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        
        self.attention_window = []
        
        while self.cap.isOpened() and self.is_in_attention_detection_mode:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame. Stopping attention detection.")
                break
                
            # Process frame
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            # Update attention window
            current_time = time()
            self.attention_window.append((current_time, attention))
            
            # Calculate metrics
            metrics = calculate_attention_metrics(self.attention_window)
            
            # Update shared variables
            with self.gaze_score_lock:
                self.gaze_score = metrics["gaze_score"]
                self.visualisation_frame = frame
            
            with self.robot_looks_lock:
                self.robot_looks = metrics["robot_looks"]
            
            with self.gaze_entropy_lock:
                self.gaze_entropy = metrics["gaze_entropy"]
            
            # Add metrics to display
            if face_found:
                h, w, _ = frame.shape
                # Add metrics
                cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                        (20, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Score: {metrics["gaze_score"]:.1f}', 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
        self.cap.release()
        cv2.destroyAllWindows()

def save_DGEI_data(name, time_now, training_dict):
    save_path = f'{name}/{name}_{time_now}_DGEI.yaml'
    with open(save_path, 'w') as file:
        yaml.dump(training_dict, file)     
        
if __name__=="__main__":
    # Create controller with more permissive thresholds
    controller = SimpleGazeController(
        camera_id=2,
        pitch_threshold=25,      # Increased from 15
        yaw_threshold=30,        # Increased from 20
        vertical_allowance=10    # Additional downward tolerance
    )
    
    # Start detection immediately (no calibration needed)
    controller.start_detecting_attention()
    
    start_time = time()
    duration = 5 * 60  # 5 minutes in seconds
    interval = 5  # Interval in seconds
    next_print_time = start_time + interval
    save_dictionary = {}
    time_step_count = 0
    time_now = strftime('%H:%M:%S', localtime(start_time))

    if not os.path.exists('saved_data'):
        print('we are executing a new session')
        os.makedirs('saved_data')
        print('made directory:' + 'saved_data' + ' for testing data')


    try:
        while time() - start_time < duration:            
            # Print the gaze score every 5 seconds
            current_time = time()
            if current_time >= next_print_time:
                gaze_score = controller.get_gaze_score()
                robot_looks = controller.get_robot_looks()
                gaze_entropy = controller.get_gaze_entropy()
                print(f"####### Gaze Score: {gaze_score}")
                print(f"Robot looks: {robot_looks}")
                print(f"Gaze entropy: {gaze_entropy}")
                save_dictionary['gaze_score_timestep_'+str(time_step_count)] = gaze_score
                save_dictionary['robot_looks_timestep_'+str(time_step_count)] = robot_looks
                save_dictionary['gaze_entropy_timestep_'+str(time_step_count)] = gaze_entropy
                time_step_count += 1
                next_print_time = current_time + interval

            save_DGEI_data('saved_data', time_now, save_dictionary)
            
            frame = controller.get_visualisation_frame()
            if frame is not None:
                cv2.imshow('Simple Gaze Detection', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            sleep(0.05)
        
        cv2.destroyAllWindows()
        controller.kill_attention_thread()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        controller.kill_attention_thread()
    
    print("Attention detection completed.")