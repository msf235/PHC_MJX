import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


from phc_mjx.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    else:
        print("not mapped", chr(keycode))
    
    
        
if __name__ == "__main__":
    device = torch.device("cpu")
    humanoid_xml = "phc_mjx/data/robot/bd/atlas_nub_hand.xml"
    sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
    # motion_file = "data/talos/0-KIT_3_walking_slow08_poses.pkl"
    # motion_file = "data/h1/gestures_4.pkl"
    # motion_data = joblib.load(motion_file)
    # motion_data_keys = list(motion_data.keys())
    # curr_motion_key = motion_data_keys[motion_id]
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    
    RECORDING = False
    
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        # for _ in range(24):
        #     add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()
             
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # pose_aa_h1_new = torch.cat([torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()[None, :, None]), H1_ROTATION_AXIS * dof_pos[None, ..., None], torch.zeros((1, 1, 2, 3))], axis = 2).float()
            # for i in range(rg_pos_t.shape[1]):
            #     if not i in [20, 21, 22]:
            #         continue
            #     viewer.user_scn.geoms[i].pos = rg_pos_t[0, i]
                # viewer.user_scn.geoms[i].pos = mj_data.xpos[:][i]
            
            # joint_gt = motion_data[curr_motion_key]['smpl_joints']
            
            # for i in range(joint_gt.shape[1]):
                # viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
                
            # joints_opt = kps_dict['j3d_h1'].squeeze()#[h1_joint_pick_idx]
            # for i in range(len(joints_opt)):
            #     viewer.user_scn.geoms[i].pos = joints_opt[i]
            
            torque = np.random.rand(mj_model.nu)/10
            
            mj_data.ctrl[:] = torque
            mujoco.mj_step(mj_model, mj_data)
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # if RECORDING and time_step > motion_len:
            #     curr_start += num_motions
            #     motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            #     time_step = 0