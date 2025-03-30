# python phc_mjx/run.py env.task="HumanoidIm" env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl exp_name=im_obsv2_standing env.self_obs_v=2 +env.im_obs_v=2 +env.im_reward_v=2
py phc_mjx/run.py env="env_im" exp_name=im_obsv2_standing \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl
