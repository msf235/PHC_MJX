import torch
import glob
from phc_mjx.smpllib.motion_lib_base import MotionLibBase, FixHeightMode
import os.path as osp
import joblib

class MotionLibMujoco(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[str(gender.int().item())]
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
            
            
            if fix_height_mode == FixHeightMode.ankle_fix:
                height_tolorance = -0.025
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
                diff_fix = (vertices_curr[:, pick][:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                height_tolorance = 0.0
                diff_fix = (vertices_curr [:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            
            trans[..., -1] -= diff_fix
            return trans, diff_fix

    def load_data(self, motion_file, min_length=-1):
        if osp.isfile(motion_file):
            self.mode = 'file'
            self._motion_data_load = joblib.load(motion_file)
        assert len(self._motion_data_load) > 0, f"Failed to load motion data from {motion_file}"
        
        data_list = self._motion_data_load
        self._motion_data_list = data_list
        self._motion_data_keys = ['data']

        # if self.mode == MotionlibMode.file:
            # if min_length != -1:
                # data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_aa']) >= min_length}
            # else:
                # data_list = self._motion_data_load

            # self._motion_data_list = np.array(list(data_list.values()))
            # self._motion_data_keys = np.array(list(data_list.keys()))
        # else:
            # self._motion_data_list = np.array(self._motion_data_load)
            # self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        # if self.mode == MotionlibMode.directory:
            # self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
