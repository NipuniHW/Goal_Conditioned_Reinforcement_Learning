#!/usr/bin/python3.10

from configurations import GAZE_RANGES, GOAL_NAMES

def calculate_reward(current_gaze: float, goal_id: int, prev_gaze: float) -> float:
    """Calculate reward based on gaze improvement toward target range."""
    target_min, target_max = GAZE_RANGES[goal_id]
    target_center = (target_min + target_max) / 2
    
    # Check if current gaze is within target range
    if target_min <= current_gaze <= target_max:
        return 50  # Same as your original functions
    
    # Calculate distance to target center (matching your original logic)
    prev_distance = abs(prev_gaze - target_center)
    current_distance = abs(current_gaze - target_center)
    
    difference = prev_distance - current_distance
    
    if difference > 0:
        # Moving closer to target center - positive reward
        return abs(difference)/2
    elif difference < 0:
        # Moving away from target center - negative reward
        return -abs(difference)
    else:
        # No change in distance - small negative reward
        return -2

# Test cases
def test_reward_function():
    print("Testing reward function:")
    
    # Test case from your example
    result = calculate_reward(28.16, 1, 0)
    print(f"current_gaze=28.16, goal_id=1 (medium), prev_gaze=0: {result}")
    
    # Additional test cases
    test_cases = [
        (50, 1, 0),    # Moving toward medium range center
        (34, 1, 0),    # Reaching medium range boundary
        (70, 1, 50),   # Moving away from medium range
        (20, 0, 10),   # Moving within low range
        (80, 2, 90),   # Moving within high range
        (50, 1, 50),   # No movement in medium range
    ]
    
    for current, goal, prev in test_cases:
        reward = calculate_reward(current, goal, prev)
        goal_name = GOAL_NAMES[goal]
        range_info = GAZE_RANGES[goal]
        print(f"current={current}, goal={goal_name} {range_info}, prev={prev}: {reward}")

if __name__ == "__main__":
    test_reward_function()

# def calculate_reward(current_gaze: float, goal_id: int, prev_gaze: float) -> float:
#         """Calculate reward based on gaze improvement toward target."""
#         target_min, target_max = GAZE_RANGES[goal_id]
#         target_center = (target_min + target_max) / 2
        
#         previous_distance_to_goal_state = abs(prev_gaze - target_center)
#         next_distance_to_goal_state = abs(current_gaze - target_center)
#         difference = previous_distance_to_goal_state - next_distance_to_goal_state

#         # Distance-based reward component
#         current_distance = abs(current_gaze - target_center)
#         prev_distance = abs(prev_gaze - target_center)
#         distance_reward = (prev_distance - current_distance) / 100.0
        
#         difference = previous_distance_to_goal_state - next_distance_to_goal_state
#         # if the current state is not within the threshold
#         if not target_min <= current_gaze <= target_max:    
#                 if previous_distance_to_goal_state < next_distance_to_goal_state:
#                 # This is bad because the agent is moving away from the goal in the next step
#                         return -(abs(difference))
#                 elif previous_distance_to_goal_state > next_distance_to_goal_state:
#                 # This is good because the agent is moving towards the goal in the next step
#                         return abs(difference)
#                 else:
#                 # They're equal which is also bad
#                         return -2
#         else:
#                 return 50

        # # Range achievement bonus
        # # range_reward = 10.0 if target_min <= current_gaze <= target_max else 0.0
        
        # return distance_reward #+ range_reward
    
# # Test the reward function
# if __name__ == "__main__":
#     # Example usage
#     current_gaze = 28.16
#     goal_id = 1
#     prev_gaze = 0
    
#     reward = calculate_reward(current_gaze, goal_id, prev_gaze)
#     print(f"Calculated Reward: {reward}")