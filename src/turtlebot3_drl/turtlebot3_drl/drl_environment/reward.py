import math
from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM

goal_dist_initial = 0

reward_function_internal = None

def get_reward(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance):
    return reward_function_internal(succeed, action_linear, action_angular, distance_to_goal, goal_angle, min_obstacle_distance)

def get_reward_A(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
        # [-3.14, 0]
        r_yaw = -1 * abs(goal_angle)

        # [-4, 0]
        r_vangular = -1 * (action_angular**2)

        # [-1, 1]
        r_distance = (2 * goal_dist_initial) / (goal_dist_initial + goal_dist) - 1

        # [-20, 0]
        if min_obstacle_dist < 0.22:
            r_obstacle = -20
        else:
            r_obstacle = 0

        # [-2 * (2.2^2), 0]
        r_vlinear = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = r_yaw + r_distance + r_obstacle + r_vlinear + r_vangular - 1

        if succeed == SUCCESS:
            reward += 250
        elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
            reward -= 200
        return float(reward)

# Define your own reward function by defining a new function: 'get_reward_X'
# Replace X with your reward function name and configure it in settings.py

def get_reward_B(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
    global goal_dist_previous
    global env_d_max
    global prev_action_linear
    global prev_action_angular

    # --- TUNABLE WEIGHTS FOR V3 ---
    w_smooth = 1.0
    w_ttc = 1.0
    T_safe = 1.0  # seconds
    # ------------------------------

    # 1. Orientation Reward
    theta = abs(math.degrees(goal_angle))
    scaled_theta = abs((90.0 - theta) / 90.0)
    
    if theta <= 90: # Towards Goal Zone
        r_orientation = scaled_theta * -0.1
    else:           # Away from Goal Zone
        r_orientation = scaled_theta * -0.3

    # 2. Distance Reward
    scaled_d = abs(goal_dist / env_d_max)
    
    if goal_dist_previous >= goal_dist: # Towards Goal
        r_distance = scaled_d * -0.05
    else:                               # Away from Goal
        r_distance = scaled_d * -0.2

    # 3. Same State (Stuck) Penalty
    r_same_state = 0.0
    if abs(goal_dist_previous - goal_dist) < 0.001:
        r_same_state = -0.65

    # Update previous distance for the next step
    goal_dist_previous = goal_dist

    # 4. Smoothness Penalty (v3)
    delta_v = action_linear - prev_action_linear
    delta_w = action_angular - prev_action_angular
    r_smooth = -w_smooth * ((delta_v ** 2) + (delta_w ** 2))
    
    # Update previous actions for the next step
    prev_action_linear = action_linear
    prev_action_angular = action_angular

    # 5. Time-to-Collision (TTC) Penalty (v3)
    # Estimate TTC based on current linear velocity and distance to nearest obstacle
    if action_linear > 0.001:
        ttc = min_obstacle_dist / action_linear
    else:
        ttc = float('inf') # Not moving forward, TTC is effectively infinite
        
    if ttc > T_safe:
        p_ttc = 0.0
    elif 0 < ttc <= T_safe:
        p_ttc = (T_safe - ttc) / T_safe
    else: # ttc <= 0
        p_ttc = 1.0
        
    r_ttc = -w_ttc * p_ttc

    # 6. Total Reward Calculation
    r_step = -0.05
    reward = r_step + r_same_state + r_orientation + r_distance + r_smooth + r_ttc

    if succeed == SUCCESS:
        reward += 10.0
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 0.75

    return float(reward)

def get_reward_C(succeed, action_linear, action_angular, goal_dist, goal_angle, min_obstacle_dist):
    # Placeholder for your third reward function
    reward = 0.0
    
    if succeed == SUCCESS:
        reward += 10.0
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward -= 0.75
        
    return float(reward)

def reward_initalize(init_distance_to_goal, d_max):
    global goal_dist_initial
    global goal_dist_previous
    global env_d_max
    global prev_action_linear
    global prev_action_angular
    
    goal_dist_initial = init_distance_to_goal
    goal_dist_previous = init_distance_to_goal
    env_d_max = d_max
    prev_action_linear = 0.0
    prev_action_angular = 0.0

function_name = "get_reward_" + REWARD_FUNCTION
reward_function_internal = globals()[function_name]
if reward_function_internal == None:
    quit(f"Error: reward function {function_name} does not exist")
