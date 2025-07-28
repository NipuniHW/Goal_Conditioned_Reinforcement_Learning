#!/usr/bin/python3.10

import sys
import os
import cv2
from time import sleep, time
from copy import deepcopy
from pepper import Pepper
from gaze_controller import *
from experiment_functions import *
import pandas as pd

def _normalize_gaze_to_category(gaze_value: float) -> int:
    if gaze_value <= 33:
        return 0
    elif gaze_value <= 66:
        return 1
    else:
        return 2

def test_q_learning(HM1, N1, G1, V1, subject_count):
    pepper = Pepper()
    pepper.connect("172.20.10.4", 9559)
    
    if not pepper.is_connected:
        sys.exit(1)

    controller = GazeInterfaceController(camera_id=0)
    sleep(1)

    print('Press Enter to start the calibration')
    input()
    controller.calibration_exe()
    controller.start_detecting_attention()
    sleep(1)

    print('Press Enter to start the testing')
    input()

    save_dictionary = {}
    time_step_count = 0
    head_mov, nav, ges, vol = HM1, N1, G1, V1

    # === PHASE 1: Run each goal ID (0, 1, 2) for 60 seconds ===
    for goal_id in [0, 1, 2]:
        print(f"\n--- Running Goal ID {goal_id} ---\n")
        start_time = time()

        while time() - start_time < 60:
            frame = controller.get_visualisation_frame()
            if frame is not None:
                f = deepcopy(frame)
                cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            if int(time() - start_time) % 3 == 0:
                gaze_score = controller.get_gaze_score()
                state = _normalize_gaze_to_category(gaze_score)
                print(f"[Goal {goal_id}] Gaze score: {gaze_score} -> State: {state}")

                dh, dn, dg, dv = select_action(goal_id, state)
                print(f"Selected Action: {(dh, dn, dg, dv)}")

                head_mov, nav, ges, vol = pepper.update_behavior(dh, dn, dg, dv, head_mov, nav, ges, vol)

                next_gaze_score = controller.get_gaze_score()
                next_state = _normalize_gaze_to_category(next_gaze_score)

                save_dictionary[f'prev_state_subject_{subject_count}_timestep_{time_step_count}'] = state
                save_dictionary[f'prev_gaze_subject_{subject_count}_timestep_{time_step_count}'] = gaze_score
                save_dictionary[f'next_state_subject_{subject_count}_timestep_{time_step_count}'] = next_state
                save_dictionary[f'next_gaze_subject_{subject_count}_timestep_{time_step_count}'] = next_gaze_score
                save_dictionary[f'action_subject_{subject_count}_timestep_{time_step_count}'] = (dh, dn, dg, dv)

                time_step_count += 1
                sleep(0.3)

    # === PHASE 2: Run 'random_actions' for 180 seconds ===
    print("\n--- Running Pepper's random_actions animation ---\n")
    pepper.execute_animation('random_actions')
    start_time = time()

    while time() - start_time < 180:
        frame = controller.get_visualisation_frame()
        if frame is not None:
            f = deepcopy(frame)
            cv2.imshow('Calibrated HRI Attention Detection', f)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        if int(time() - start_time) % 3 == 0:
            gaze_score = controller.get_gaze_score()
            state = _normalize_gaze_to_category(gaze_score)

            save_dictionary[f'random_state_subject_{subject_count}_timestep_{time_step_count}'] = state
            save_dictionary[f'random_gaze_subject_{subject_count}_timestep_{time_step_count}'] = gaze_score

            time_step_count += 1
            sleep(0.3)

    # === Save all ===
    save_trajectory_ep_to_yaml('dynamic_experiment_data', subject_count, save_dictionary)
    controller.kill_attention_thread()
    cv2.destroyAllWindows()
    del pepper
    print("Experiment completed\n")

if __name__ == "__main__":
    initial_HM = 0
    initial_N = 0
    initial_G = 5
    initial_V = 5
    subject_count = 0    

    if not os.path.exists('dynamic_experiment_data'):
        os.makedirs('dynamic_experiment_data')
        print('Created directory: dynamic_experiment_data')

    while True:
        subject_count += 1
        user_input = input('\nStart/continue experiment for another subject? (Y/N): ')
        if user_input.lower() == 'y':
            test_q_learning(initial_HM, initial_N, initial_G, initial_V, subject_count)
        else:
            print('Exiting experiment.')
            break
