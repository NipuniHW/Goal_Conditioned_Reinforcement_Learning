#!/usr/bin/python3.10

# This is the experiment script for testing the GCRL algorithm with gaze control on Pepper robot.

import sys
import cv2
from time import sleep, time
from pepper import Pepper
from gaze_controller import *
from experiment_functions import *
from gcrl_train import GazeGCRL

def test_q_learning(duration_minutes, goal_id, HM1, N1, G1, V1, subject_count):
    pepper = Pepper()
    pepper.connect("172.20.10.4", 9559)
    # pepper.connect("localhost", 39403)
    
    try:
        if not pepper.is_connected:
            sys.exit(1)
            del pepper 
        else:
            # Change the camera ID to 2 if using external USB webcam, 0 if using the laptop webcam
            controller = GazeInterfaceController(camera_id=2)
            sleep(1)
            
            print('Press Enter to start the calibration')
            input()
            controller.calibration_exe()
            controller.start_detecting_attention()

            sleep(1)
            print('Press Enter to start the testing')
            input()
            
            current_time = time()
            save_dictionary = {}
            time_step_count = 0
            
            head_mov, nav, ges, vol = HM1, N1, G1, V1
            experiment_duration_seconds = duration_minutes * 60
            
            start_time_inner_loop = time()
            
            while time() - current_time < experiment_duration_seconds:
                frame = controller.get_visualisation_frame()
                if frame is not None:
                    f = deepcopy(frame)
                    cv2.imshow('Calibrated HRI Attention Detection', f)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                    
                if time() - start_time_inner_loop >= 3:
                    start_time_inner_loop = time()
                    gaze_score = controller.get_gaze_score()
                    state = _normalize_gaze_to_category( gaze_score)
                    print(f"Gaze score: {gaze_score} -- giving state: {state}")
                    (dh, dn, dg, dv) = select_action(goal_id, state)
                    print(f"Chosen action: {dh, dn, dg, dv}")
                    head_mov, nav, ges, vol = pepper.update_behavior(dh, dn, dg, dv, head_mov, nav, ges, vol)
                    print("Updated the behavior\n")
                    next_gaze_score = controller.get_gaze_score()
                    next_state = _normalize_gaze_to_category(next_gaze_score)

                    save_dictionary['previousstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = state
                    save_dictionary['previousstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = gaze_score
                    save_dictionary['nextstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = next_state
                    save_dictionary['previousstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = next_gaze_score
                    save_dictionary['action_subject_(dh,dn,dg,dv)' + str(subject_count)+'_timestep_'+str(time_step_count)] = int(dh), int(dn), int(dg), int(dv)

                    time_step_count+=1
            
                sleep(0.18)
            save_trajectory_ep_to_yaml('experiment_data', subject_count, save_dictionary)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del pepper   
        cv2.destroyAllWindows()
        sys.exit(0)

    del pepper 
    controller.kill_attention_thread()
    print("Experiment completed\n")

def _normalize_gaze_to_category(gaze_value: float) -> int:
    """Convert continuous gaze value to categorical (0, 1, 2)."""
    if gaze_value <= 33:
        return 0
    elif gaze_value <= 66:
        return 1
    else:
        return 2
        
if __name__ == "__main__":
    duration_minutes = 1            # Duration of the test in minutes
    goal_id = 0                      # Goal ID for the test
    initial_HM = 0                   # Initial head movement level
    initial_N = 0                    # Initial navigation level
    initial_G = 5                    # Initial gesture level
    initial_V = 5                    # Initial volume level
    subject_count = 0    

    if not os.path.exists('experiment_data'):
            print('we are executing a new testing session')
            os.makedirs('experiment_data')
            print('made directory:' + 'experiment_data' + ' for testing data')
        
    while True:
        subject_count += 1
        user_input = input('\nWould you like to start/ continue experiment for another subject? (Y/N): ')
        if user_input.lower() == 'y' or user_input.lower() == 'Y':
            test_q_learning(duration_minutes, goal_id, initial_HM, initial_N, initial_G, initial_V, subject_count)
        else:
            print('Your input was not Y/y. Exiting experiment')
            break
