#!/usr/bin/python3.10

# import time
# import cv2
# import torch
# from copy import deepcopy

# from gcrl_train import GazeGCRL              # Import your training class
# from entropy_interfacing import GazeInterfaceController
# from pepper import Pepper                   # Interface for Pepper robot

# # Path to your trained model
# MODEL_PATH = "/home/vscode/gaze_ws/L2CS-Net/goal_conditioned_model.pth"

# # Target goal level (0 = Low, 1 = Medium, 2 = High)
# GOAL_ID = 0

# def main():
#     # Load trained model
#     print("Loading trained model...")
#     gcrl_model = GazeGCRL()
#     gcrl_model.load_model(MODEL_PATH)
#     gcrl_model.epsilon = 0.0  # Turn off exploration

#     # Connect to Pepper
#     print("Connecting to Pepper...")
#     pepper = Pepper()
#     # pepper.connect("172.20.10.4", 9559)     # Physical Pepper IP
#     # pepper.connect("localhost", 44889)      # Virtual Pepper IP

#     # Initialize gaze detection
#     print("Starting gaze detection...")
#     controller = GazeInterfaceController(camera_id=0)
#     input("Press Enter to start calibration...")
#     controller.calibration_exe()
#     controller.start_detecting_attention()

#     print("Testing trained model in real time. Press ESC to exit.")

#     # Initial Pepper behavior state
#     current_head_mov = 2
#     current_navigation = 4
#     current_gesture = 5
#     current_volume = 5

#     try:
#         while True:
#             # Get frame for visualization
#             frame = controller.get_visualisation_frame()
#             if frame is not None:
#                 cv2.imshow("Real-Time Gaze", frame)
#                 if cv2.waitKey(5) & 0xFF == 27:
#                     break

#             # Get gaze score and update state
#             current_gaze = controller.get_gaze_score()
#             print(f"Gaze score: {current_gaze:.2f}")
#             state = gcrl_model.update_state(current_gaze)

#             # Get best action (greedy policy)
#             _, dh, dn, dg, dv = gcrl_model.select_action(state, GOAL_ID, training=False)
#             print(f"Selected action: Δhead={dh}, Δnav={dn}, Δgesture={dg}, Δvolume={dv}")

#             # Update Pepper's behavior
#             current_head_mov, current_navigation, current_gesture, current_volume = pepper.update_behavior(
#                 dh, dn, dg, dv,
#                 current_head_mov, current_navigation, current_gesture, current_volume
#             )

#             time.sleep(3)  # Allow time to observe behavior/gaze change

#     finally:
#         print("Stopping gaze detection and closing...")
#         controller.kill_attention_thread()
#         cv2.destroyAllWindows()
#         del pepper

# if __name__ == "__main__":
#     main()

