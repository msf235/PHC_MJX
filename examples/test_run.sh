# On branch small-fixes
python phc_mjx/run.py env=env_im env.motion_file=data/amass/amass_isaac_standing_upright_slim.pkl exp_name=im_obsv2_1 env.self_obs_v=2 headless=True test=False num_threads=1 learning.min_batch_size=256

py phc_mjx/run.py env=env_im env.motion_file=data/sim_data.pkl exp_name=im_obsv2_1 env.self_obs_v=2 headless=True test=False num_threads=1 learning.min_batch_size=256 env.motion_file_type='mujoco' env.model_file="data/mujoco/humanoid_and_tennis.xml" robot.humanoid_type="smpl" robot.xml_file=""data/mujoco/humanoid_and_tennis.xml env.reset_bodies_file="" robot.use_smpl_data="False"
