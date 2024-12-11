import torch
import numpy as np
import glob
from phc_mjx.smpllib.motion_lib_base import MotionLibBase, FixHeightMode
import os.path as osp
import joblib
import mujoco as mj
import torch.multiprocessing as mp

class MotionLibMujoco(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)
        model = mj.MjModel.from_xml_path(motion_lib_cfg.model_file)
        self.num_joints = model.njnt
        # TODO: need to load mujoco xml file info
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
        self._motion_data_keys = np.array(['data'])

        # TODO: integrate this logic in new code
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
        
        data_list = self._motion_data_load
        self._motion_data_list = np.array(list(data_list.values()))
        self._motion_data_keys = np.array(list(data_list.keys()))
        self._num_unique_motions = len(self._motion_data_list)
        # if self.mode == MotionlibMode.directory:
            # self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
#         breakpoint()

    def load_motions(self, m_cfg, shape_params, random_sample=True, start_idx=0, silent= False):
        # load motion load the same number of motions as there are skeletons (humanoids)

        motions = []
        motion_lengths = []
        motion_fps_acc = []
        motion_dt = []
        motion_num_frames = []
        # motion_bodies = []
        # motion_aa = []

        total_len = 0.0
        
        # self.num_joints = len(self.mesh_parsers["0"].joint_names)
        num_motion_to_load = len(shape_params)
        if random_sample:
            sample_idxes = np.random.choice(np.arange(self._num_unique_motions), size = num_motion_to_load, p = self._sampling_prob, replace=True)
        else:
            sample_idxes = np.remainder(np.arange(num_motion_to_load) + start_idx, self._num_unique_motions )

#         breakpoint()

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        motion_data_list = self._motion_data_list[sample_idxes]
        self.qpos = motion_data_list[0]['qs']
        self.qvel = motion_data_list[0]['vs']
        self.gts  = motion_data_list[0]['xpos']
        self.grs  = motion_data_list[0]['xquats']
        self.dt   = motion_data_list[0]['dt']
        self._motion_dt = np.array([self.dt]).astype(self.dtype)

        num_frames = self.gts.shape[0]

        # motion_fps = curr_motion.fps
        # curr_dt = 1.0 / motion_fps
        # num_frames = curr_motion.global_translation.shape[0]
        # curr_len = 1.0 / motion_fps * (num_frames - 1)
        curr_len = self.dt * (num_frames - 1)
        self._motion_lengths = np.array([curr_len]).astype(self.dtype)
        self.num_bodies = motion_data_list[0]['njnts']
        self._motion_num_frames = np.array([num_frames])

        lengths = self._motion_num_frames
        lengths_shifted = np.roll(lengths, 1, axis = 0) 
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

#         breakpoint()

        # mp.set_sharing_strategy('file_descriptor')

        # manager = mp.Manager()
        # queue = manager.Queue()
        # num_jobs = min(min(mp.cpu_count(), 64), num_motion_to_load)
        # jobs = motion_data_list
        
        # if len(jobs) <= 32 or not self.multi_thread or num_jobs <= 8:
            # num_jobs = 1
            
        # res_acc = {}  # using dictionary ensures order of the results.
        
        # chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        # ids = np.arange(len(jobs))

        # jobs = [(ids[i:i + chunk], jobs[i:i + chunk], shape_params[i:i + chunk], self.mesh_parsers, m_cfg) for i in range(0, len(jobs), chunk)]
        # job_args = [jobs[i] for i in range(len(jobs))]
        # for i in range(1, len(jobs)):
            # worker_args = (*job_args[i], queue, i)
            # worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            # worker.start()
        # # res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))
        # # pbar = tqdm(range(len(jobs) - 1)) if not silent else range(len(jobs) - 1)
        # # for i in pbar:
            # # res = queue.get()
            # # res_acc.update(res)
        # # pbar = tqdm(range(len(res_acc)))if not silent else range(len(res_acc))
                                                                 
        # for f in pbar:
            # motion_file_data, curr_motion = res_acc[f]

            # motion_fps = curr_motion.fps
            # curr_dt = 1.0 / motion_fps

            # num_frames = curr_motion.global_translation.shape[0]
            # curr_len = 1.0 / motion_fps * (num_frames - 1)
            
            # # motion_aa.append(curr_motion.pose_aa)
            # # motion_bodies.append(curr_motion.gender_beta)

            # motion_fps_acc.append(motion_fps)
            # motion_dt.append(curr_dt)
            # motion_num_frames.append(num_frames)
            # motions.append(curr_motion)
            # motion_lengths.append(curr_len)
            
            # del curr_motion
            
        # self._motion_lengths = np.array(motion_lengths).astype(self.dtype)
        # self._motion_fps = np.array(motion_fps).astype(self.dtype)
        # # self._motion_bodies = np.stack(motion_bodies).astype(self.dtype)
        # # self._motion_aa = np.concatenate(motion_aa).astype(self.dtype)

        # self._motion_dt = np.array(motion_dt).astype(self.dtype)
        # self._motion_num_frames = np.array(motion_num_frames)
        # self._num_motions = len(motions)

        # # self.gts = np.concatenate([m.global_translation for m in motions], axis=0).astype(self.dtype)
        # # self.grs = np.concatenate([m.global_rotation for m in motions], axis=0).astype(self.dtype)
        # # self.lrs = np.concatenate([m.local_rotation for m in motions], axis=0).astype(self.dtype)
        # # self.grvs = np.concatenate([m.global_root_velocity for m in motions], axis=0).astype(self.dtype)
        # # self.gravs = np.concatenate([m.global_root_angular_velocity for m in motions], axis=0).astype(self.dtype)
        # # self.gavs = np.concatenate([m.global_angular_velocity for m in motions], axis=0).astype(self.dtype)
        # # self.gvs = np.concatenate([m.global_velocity for m in motions], axis=0).astype(self.dtype)
        # # self.dvs = np.concatenate([m.dof_vels for m in motions], axis=0).astype(self.dtype)
        # # self.dof_pos = np.concatenate([m.dof_pos for m in motions], axis=0).astype(self.dtype)
        # self.qpos = np.concatenate([m.qpos for m in motions], axis=0).astype(self.dtype)
        # self.qvel = np.concatenate([m.qvel for m in motions], axis=0).astype(self.dtype)
        
        # lengths = self._motion_num_frames
        # lengths_shifted = np.roll(lengths, 1, axis = 0) 
        # lengths_shifted[0] = 0
        # self.length_starts = lengths_shifted.cumsum(0)
        # self.motion_ids = np.arange(len(motions))
        # motion = motions[0]
        # self.num_bodies = self.num_joints

        # num_motions = self.num_current_motions()
        # total_len = self.get_total_length()
        # if not silent:
            # print(f"###### Sampling {num_motions:d} motions:", sample_idxes[:5], self.curr_motion_keys[:5], f"total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        # else:
            # print(sample_idxes[:5], end=" ")
        # return motions
