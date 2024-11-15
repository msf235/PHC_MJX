

MUJOCO_GAINS = {}
MUJOCO_GAINS["uhc_pd"] = {
    # # Much bigger
    "hip_x_left":       [800, 80, 1, 1000],
    "knee_left":        [800, 80, 1, 1000],
    "knee_right":       [800, 80, 1, 1000],
    "hip_x_right":      [800, 80, 1, 1000],
    "knee_right":       [800, 80, 1, 1000],
    "abdomen_x":        [1000, 100, 1, 500],
    "shoulder1_left":  [500, 50, 1, 1000],
    "elbow_left":       [500, 50, 1, 250],
    "shoulder1_right": [500, 50, 1, 1000],
    "elbow_right":      [500, 50, 1, 250],
    "wrist_right":      [300, 30, 1, 250],
    "wrist_left":       [300, 30, 1, 250],
}

# simple pd:
MUJOCO_GAINS["simplepd"] = {
    "hip_x_left":       [250, 5, 1, 500, 10, 2],
    "knee_left":        [250, 5, 1, 500, 10, 2],
    "knee_right":       [150, 5, 1, 500, 10, 2],
    "hip_x_right":      [150, 3, 1, 500, 1, 1],
    "knee_right":       [250, 5, 1, 500, 10, 2],
    "abdomen_x":        [250, 5, 1, 500, 10, 2],
    "shoulder1_left":  [150, 5, 1, 500, 10, 2],
    "elbow_left":       [150, 3, 1, 500, 1, 1],
    "shoulder1_right": [500, 10, 1, 500, 10, 2],
    "elbow_right":      [500, 10, 1, 500, 10, 2],
    "wrist_right":      [500, 10, 1, 500, 10, 2],
    "wrist_left":       [150, 1, 1, 250, 50, 4],
}
