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
    """
    Reward Function v3 (Updated)
    
    Total Reward:
      R = w_g*R_goal + w_c*R_collision + w_s*R_step + w_o*R_orient
        + w_d*R_distance + w_same*R_same
        - w_ttc*P_ttc - w_smooth*[(Δv)² + (Δω)²]
        + R_obstacle_proximity + R_timeout
    """
    global goal_dist_previous
    global env_d_max
    global prev_action_linear
    global prev_action_angular

    # ===================== TUNABLE WEIGHTS (v3) ===================== #
    w_g      = 10.0     # Goal weight:       base +10  × 10  = +100 effective
    w_c      = 200.0    # Collision weight:   base -0.75× 200 = -150 effective
    w_s      = 1.0      # Step weight:        base -0.05× 1   = -0.05 effective
    w_o      = 1.0      # Orientation weight
    w_d      = 1.0      # Distance weight
    w_same   = 1.0      # Same-state weight
    w_ttc    = 1.0      # TTC weight
    w_smooth = 0.1      # Smoothness weight (kept low to allow corrective turns)
    T_safe   = 1.0      # TTC safe threshold in seconds
    w_timeout = 25.0    # Timeout penalty (not in original v3, added for convergence)
    # ================================================================ #

    # ---------- BASE REWARD VALUES (from v3 diagram) ----------

    # R_goal and R_collision (applied at episode end)
    R_goal      = 10.0
    R_collision = -0.75

    # R_step: small per-step penalty to encourage faster completion
    R_step = -0.05

    # R_orient: orientation reward based on angle to goal
    theta = abs(math.degrees(goal_angle))       # Range: [0°, 180°]
    scaled_theta = abs((90.0 - theta) / 90.0)   # Scaled to [0, 1]

    if theta <= 90:  # Facing towards goal zone
        R_orient = scaled_theta * -0.1
    else:            # Facing away from goal zone
        R_orient = scaled_theta * -0.3

    # R_distance: distance shaping based on movement towards/away from goal
    scaled_d = abs(goal_dist / env_d_max)       # Scaled to [0, 1]

    if goal_dist_previous >= goal_dist:  # Moved closer to goal
        R_distance = scaled_d * -0.05
    else:                                # Moved away from goal
        R_distance = scaled_d * -0.2

    # R_same: penalty if robot remains in the same state (no progress)
    R_same = 0.0
    if abs(goal_dist_previous - goal_dist) < 0.001:  # Less than 1mm movement
        R_same = -0.65

    # Update distance tracker for next step
    goal_dist_previous = goal_dist

    # ---------- v3 ADDITIONS ----------

    # Smoothness Penalty: penalizes sudden changes in velocity
    delta_v = action_linear - prev_action_linear
    delta_w = action_angular - prev_action_angular
    R_smooth = (delta_v ** 2) + (delta_w ** 2)

    # Update action trackers for next step
    prev_action_linear = action_linear
    prev_action_angular = action_angular

    # Time-to-Collision (TTC) Penalty: penalizes when collision is imminent
    if action_linear > 0.001:
        ttc = min_obstacle_dist / action_linear
    else:
        ttc = float('inf')  # Not moving forward, no collision risk

    if ttc > T_safe:
        P_ttc = 0.0
    elif 0 < ttc <= T_safe:
        P_ttc = (T_safe - ttc) / T_safe
    else:  # ttc <= 0
        P_ttc = 1.0

    # ---------- ADDITIONAL FIXES (not in original v3) ----------

    # Obstacle Proximity Penalty: continuous danger-zone warning
    # Gives gradient signal to steer away BEFORE collision occurs
    DANGER_THRESHOLD   = 0.50   # meters: start gentle warning
    CRITICAL_THRESHOLD = 0.25   # meters: harsh penalty zone
    R_obstacle_prox = 0.0
    if min_obstacle_dist < DANGER_THRESHOLD:
        if min_obstacle_dist < CRITICAL_THRESHOLD:
            # Very close: harsh penalty, range [-20, -5]
            R_obstacle_prox = -20.0 * (1.0 - min_obstacle_dist / CRITICAL_THRESHOLD)
        else:
            # Approaching: gentle warning, range [-5, 0]
            R_obstacle_prox = -5.0 * (1.0 - (min_obstacle_dist - CRITICAL_THRESHOLD) / (DANGER_THRESHOLD - CRITICAL_THRESHOLD))

    # ---------- TOTAL REWARD CALCULATION ----------
    # R = w_g*R_goal + w_c*R_collision + w_s*R_step + w_o*R_orient
    #   + w_d*R_distance + w_same*R_same
    #   - w_ttc*P_ttc - w_smooth*R_smooth
    #   + R_obstacle_prox + R_timeout

    reward = (w_s * R_step) + (w_o * R_orient) + (w_d * R_distance) + (w_same * R_same) \
           - (w_ttc * P_ttc) - (w_smooth * R_smooth) \
           + R_obstacle_prox

    # Terminal rewards (applied once at the end of an episode)
    if succeed == SUCCESS:
        reward += w_g * R_goal              # +100
    elif succeed == COLLISION_OBSTACLE or succeed == COLLISION_WALL:
        reward += w_c * R_collision         # -150
    elif succeed == TIMEOUT:
        reward -= w_timeout                 # -25

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
