# Pinned H3 latent-upscaler inference

`minimax_h3_latent_upscaler_2d.py` is the unchanged MIT-licensed inference
module from `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`, commit
`f34cadb19f10fee7e36091b690cac39b4eb40573`.
SHA256: `8c95ea2c09f474464640570b67b61565ac6d8aa38c579b0662a5e080e26157b8`.
Its original license is included as `MINIMAX_H3_LATENT_UPSCALER_LICENSE`.

The director imports this inference module lazily. It does not register its
node classes or alter the user's installed latent-upscaler extension.
`nodes/selflift_lifter.py` owns checkpoint identity checks and integration.
The model weight is external; it is not bundled in this repository.

Reviewed optional model: `LBH-123-AI/Minimax_h3_latent_Upscaler`, revision
`85a0187096046badc0a71ab6bd6be182ab06c0f4`, file
`minimax_h3_latent_upscaler_2d_conv_v1/minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors`.
SHA256: `6e0f35692784ba2376704a48a90fb07f90aef76ef21200d5525d6627a54fe909`.
Only tier1 of this exact checkpoint is selected. An installed older2D node
is not assumed compatible with these weights.
