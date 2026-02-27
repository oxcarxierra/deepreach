python run_experiment.py --mode train --experiment_class DeepReach \
    --dynamics_class Dubins3D --experiment_name dubins3d_tutorial_run \
    --minWith target --goalR 0.25 --velocity 0.6 --omega_max 1.1 --angle_alpha_factor 1.2 --set_mode avoid \
    --use_wandb --wandb_project deepreach --wandb_entity oxcarxierra-seoul-national-university  --wandb_group dubins3d_tutorial_run --wandb_name dubins3d_tutorial_run_2