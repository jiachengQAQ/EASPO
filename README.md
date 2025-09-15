# EAeaspo Training and Inference Code

This folder contains the code for EAeaspo training and inference.


1. Run the Docker Container and Enter It

> ```bash
> conda env create -f environment.yaml --name easpo
> conda activate easpo
> ```

2. Login to wandb
```bash
wandb login {Your wandb key}
```

## Training and Inference
sudo apt update
sudo apt install wget

mkdir model_ckpts
cd model_ckpts

```bash
To fine-tune easop with grad-tts
accelerate launch --config_file accelerate_cfg/1a4o_fp16.yaml train_scripts/train_easpo.py --config configs/easpo_-v1-5_4k-prompts_num-sam-4_10ep_bs10.py
```

```bash
To inference easop with grad-tts
python inference_scripts/inference.py
```