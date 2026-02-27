## Running TransReach (Transformer-Based DeepReach)
`run_experiment_transreach.py` implements a sequence-based TransReach model:
- pseudo-sequence generation over time (`pseudo_steps`, `pseudo_dt`)
- encoder-decoder attention over pseudo tokens
- Wavelet activation (`w1*sin(x) + w2*cos(x)`)
- sequential HJI losses integrated into the existing DeepReach training loop

Example:
```
python run_experiment_transreach.py --mode train --experiment_class DeepReach --dynamics_class Air3D --experiment_name air3d_transreach --minWith target --collisionR 0.25 --velocity 0.75 --omega_max 3.0 --angle_alpha_factor 1.2 --num_nl 256 --num_heads 2 --num_encoder_layers 1 --num_decoder_layers 1 --pseudo_steps 5 --pseudo_dt 1e-3 --device auto
```

For Ubuntu GPU sweeps, see:
`scripts/run_transreach_air3d_ubuntu.sh`

For a quick CUDA environment setup on Ubuntu:
`scripts/setup_ubuntu_cuda_env.sh`