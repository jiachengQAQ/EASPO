from functools import partial
import os
import os.path as osp
import sys
# Add the project directory to the Python path to simplify imports without manually setting PYTHONPATH.
sys.path.insert(
    0, osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), "..")
    ),
)
import copy
import contextlib
import math
import json

import tqdm
import torch
import wandb

from absl import app, flags
from ml_collections import config_flags
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, broadcast
from accelerate.logging import get_logger
from diffusers import GradTTSPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from peft import LoraConfig
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from spo.preference_models import get_preference_model_func, get_compare_func
from spo.datasets import build_dataset
from spo.utils import (
    huggingface_cache_dir, 
    UNET_CKPT_NAME, 
    gather_tensor_with_diff_shape,
)
from spo.custom_diffusers import (
    multi_sample_pipeline, 
    step_with_logprob,
)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", 
from model import GradTTS
from text import text_to_sequence, cmudict
from utils import intersperse
from models import Generator as HiFiGAN
from env import AttrDict
import numpy as np
from scipy.io.wavfile import write

# Initialize Grad-TTS
generator = GradTTS(...)
generator.eval().cuda()

# Initialize HiFi-GAN vocoder
with open('./checkpts/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
vocoder = HiFiGAN(h)
vocoder.load_state_dict(torch.load('./checkpts/hifigan.pt')['generator'])
vocoder.eval().cuda()
vocoder.remove_weight_norm()

    "configs/spo_-v1-5_4k-prompts_num-sam-4_10ep_bs10.py", 
    "Training configuration."
)

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config
    config = Config(config.to_dict())
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # timesteps used for training: [divert_start_step: num_sample_timesteps]
    divert_start_step = config.train.divert_start_step

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb" if not getattr(config, 'debug', False) else None,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        if not getattr(config, 'debug', False):
            accelerator.init_trackers(
                project_name=config.wandb_project_name, 
                config=config, 
                init_kwargs={"wandb": {
                    "name": config.run_name, 
                    "entity": config.wandb_entity_name
                }}
            )
        os.makedirs(os.path.join(config.logdir, config.run_name), exist_ok=True)
        with open(os.path.join(config.logdir, config.run_name, "exp_config.py"), "w") as f:
            f.write(config.pretty_text)
    logger.info(f"\n{config.pretty_text}")

    set_seed(config.seed, device_specific=True)
    
    # For mixed precision training we cast all non-trainable weigths (vocoder, text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load models.
    pipeline = GradTTSPipeline.from_pretrained(
        config.pretrained.model, 
        torch_dtype=inference_dtype,
        cache_dir=huggingface_cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained.model,
        subfolder="unet",
        cache_dir=huggingface_cache_dir,
    )
    pipeline.unet = unet
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vocoder.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if config.use_checkpointing:
        unet.enable_gradient_checkpointing()
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=2,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Sampling Timestep",
        dynamic_ncols=True,
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    
    preference_model_fn = get_preference_model_func(config.preference_model_func_cfg, accelerator.device)
    compare_func = get_compare_func(config.compare_func_cfg)

    # Move unet, vocoder and text_encoder to device and cast to inference_dtype
    pipeline.vocoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        unet.to(accelerator.device, dtype=inference_dtype)
        unet.requires_grad_(False)
    else:
        unet.requires_grad_(True)
    #### Prepare reference model
    ref = copy.deepcopy(unet)
    ref.to(accelerator.device)
    ref.requires_grad_(False)
    
    if config.use_lora:
        unet_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        if accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if config.use_lora:
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(models[0])
                )
                GradTTSPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                )
                logger.info(f"saved lora weights to {output_dir}")
            else:
                models[0].save_pretrained(os.path.join(output_dir, UNET_CKPT_NAME))
                logger.info(f"saved weights to {os.path.join(output_dir, UNET_CKPT_NAME)}")
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], type(accelerator.unwrap_model(unet))):
            if config.use_lora:
                lora_state_dict, network_alphas = GradTTSPipeline.lora_state_dict(input_dir)
                unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
                unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
                incompatible_keys = set_peft_model_state_dict(models[0], unet_state_dict, adapter_name="default")
                if incompatible_keys is not None:
                    # check only for unexpected keys
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                            f" {unexpected_keys}. "
                        )
                if accelerator.mixed_precision == "fp16":
                    # only upcast trainable parameters (LoRA) into fp32
                    cast_training_params([models[0]], dtype=torch.float32)
                logger.info(f"loaded lora weights from {input_dir}")                
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=UNET_CKPT_NAME)
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                logger.info(f"loaded weights from {input_dir}")                
                del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_para = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = optimizer_cls(
        trainable_para,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_dataset = build_dataset(config.dataset_cfg)
    collate_fn = partial(
        prompt_dataset.collate_fn,
        tokenizer=pipeline.tokenizer,
    )

    data_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        collate_fn=collate_fn,
        batch_size=config.sample.sample_batch_size,
        num_workers=config.dataloader_num_workers,
        shuffle=config.dataloader_shuffle,
        pin_memory=config.dataloader_pin_memory,
        drop_last=config.dataloader_drop_last,
    )
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    # for some reason, autocast is necessary for non-lora training but not for lora training, and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with `accelerator`.
    unet, optimizer, data_loader = accelerator.prepare(unet, optimizer, data_loader)
        
    # Train!
    total_train_batch_size = (
        config.train.train_batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sampling batch size per device = {config.sample.sample_batch_size}")
    logger.info(f"  Training batch size per device = {config.train.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        with open(os.path.join(config.resume_from, "global_step.json"), "r") as f:
            global_step = json.load(f)["global_step"]
    else:
        first_epoch = 0
        global_step = 0
    
    for epoch in tqdm(
        range(first_epoch, config.num_epochs),
        total=config.num_epochs,
        initial=first_epoch,
        disable=not accelerator.is_local_main_process,
        desc="Epoch",
        position=0,
    ):
        train_loss = 0.0
        train_ratio_win = 0.0
        train_ratio_lose = 0.0
        for dataset_batch_idx, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not accelerator.is_local_main_process,
            desc="Batch",
            position=1,
        ):
            if (
                dataset_batch_idx == len(data_loader) - 1 and 
                accelerator.gradient_state.in_dataloader
            ):
                # After sampling, we need to iterate through training batches.
                # If 'end_of_dataloader' is True, accelerator.accumulate will skip gradient accumulation.
                # Hence, we set it to False to ensure proper gradient accumulation.
                accelerator.gradient_state.active_dataloader.end_of_dataloader = False

            #################### SAMPLING ####################
            unet.eval()
            pipeline.unet.eval()
            batch_size = batch['input_ids'].shape[0]
            prompt_ids = batch['input_ids']
            # encode prompts
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            sample_neg_prompt_embeds = neg_prompt_embed.repeat(batch_size, 1, 1)
            
            # prepare extra_info for the preference model
            extra_info = batch['extra_info']
            for k, v in extra_info.items():
                if isinstance(v, torch.Tensor):
                    other_dim = [1 for _ in range(v.dim() - 1)]
                    extra_info[k] = v.repeat(config.sample.num_sample_each_step, *other_dim)
                elif isinstance(v, list):
                    extra_info[k] = v * config.sample.num_sample_each_step
                else:
                    raise ValueError(f"Unknown type {type(v)} for extra_info[{k}]")
            with autocast():
                (
                    timesteps, 
                    current_latents,  # x_t
                    next_latents, # x_{t-1}
                    prompt_embeds,
                    preference_score_logs,
                ) = multi_sample_pipeline(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    
                    divert_start_step=divert_start_step,
                    num_samples_each_step=config.sample.num_sample_each_step,
                    preference_model_fn=preference_model_fn,
                    compare_fn=compare_func,
                    extra_info=extra_info,
                )
                
            preference_score_logs = accelerator.gather(preference_score_logs).detach()
            accelerator.log(
                {
                    "preference_scores_mean": preference_score_logs.mean().item(), 
                    "preference_scores_std": preference_score_logs.std().item(),
                },
                step=global_step,
            )
            del preference_score_logs
            
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                local_valid_samples_num_list = [
                    torch.tensor([next_latents.shape[0]], dtype=torch.long, device=accelerator.device) 
                    for _ in range(accelerator.num_processes)
                ]
                for process_idx in range(accelerator.num_processes):
                    broadcast(local_valid_samples_num_list[process_idx], from_process=process_idx)
                
                local_valid_samples_num_list = [sample_num.item() for sample_num in local_valid_samples_num_list]

               
                timesteps = gather_tensor_with_diff_shape(timesteps, local_valid_samples_num_list)
                
                current_latents = gather_tensor_with_diff_shape(current_latents, local_valid_samples_num_list)
                
                next_latents = gather_tensor_with_diff_shape(next_latents, local_valid_samples_num_list)
                
                prompt_embeds = gather_tensor_with_diff_shape(prompt_embeds, local_valid_samples_num_list)
            
            total_valid_samples_num = timesteps.shape[0]
            
            if total_valid_samples_num < accelerator.num_processes:
                continue
            
            sample = {
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": current_latents,  # x_t
                "next_latents": next_latents,  # x_{t-1}
            }
            
            if accelerator.is_main_process:
                valid_perm = torch.randperm(total_valid_samples_num, device=accelerator.device)
                accelerator.wait_for_everyone()
                broadcast(valid_perm, from_process=0)
                accelerator.wait_for_everyone()
            else:
                valid_perm = torch.ones(
                    total_valid_samples_num,
                    dtype=torch.long,
                    device=accelerator.device,
                ) * -1
                accelerator.wait_for_everyone()
                broadcast(valid_perm, from_process=0)
                accelerator.wait_for_everyone()
                assert not torch.any(valid_perm == -1)
            
            num_items_per_gpu = total_valid_samples_num // accelerator.num_processes
            valid_start_index = accelerator.process_index * num_items_per_gpu
            valid_end_index = valid_start_index + num_items_per_gpu
            for key, value in sample.items():
                sample[key] = value[valid_perm]
                sample[key] = sample[key][valid_start_index: valid_end_index]
            del prompt_embeds
            del timesteps
            del current_latents
            del next_latents
            
            sample_0 = {}
            sample_1 = {}
            for key, value in sample.items():
                if value.shape[1] == 1:
                    sample_0[key] = value[:, 0]
                    sample_1[key] = value[:, 0]
                else:
                    sample_0[key] = value[:, 0]
                    sample_1[key] = value[:, 1]
            del sample
            
            torch.cuda.empty_cache()
            
            num_train_batches = math.ceil(sample_0['latents'].shape[0] / config.train.train_batch_size)
            
            ############ Training ############
            unet.train()
            pipeline.unet.train()
            for train_batch_idx in tqdm(
                range(num_train_batches),
                desc="Training Small Batches",
                position=2,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                train_b_start = config.train.train_batch_size * train_batch_idx
                train_b_end = config.train.train_batch_size * (train_batch_idx + 1)
                if config.train.cfg:
                    train_neg_prompt_embeds = neg_prompt_embed.repeat(
                        sample_0["prompt_embeds"][train_b_start: train_b_end].shape[0], 
                        1, 1,
                    )
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"][train_b_start: train_b_end]])
                    embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"][train_b_start: train_b_end]])
                else:
                    embeds_0 = sample_0["prompt_embeds"][train_b_start: train_b_end]
                    embeds_1 = sample_1["prompt_embeds"][train_b_start: train_b_end]
                
                with accelerator.accumulate(unet):
                    with autocast():
                        if config.train.cfg:
                            noise_pred_0 = unet(
                                torch.cat([sample_0["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_0["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_0,
                            ).sample
                            noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                            noise_pred_0 = noise_pred_uncond_0 + config.sample.guidance_scale * (
                                noise_pred_text_0 - noise_pred_uncond_0
                            )
                            noise_ref_pred_0 = ref(
                                torch.cat([sample_0["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_0["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_0,
                            ).sample
                            noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                            noise_ref_pred_0 = noise_ref_pred_uncond_0 + config.sample.guidance_scale * (
                                noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                            )
                            
                            noise_pred_1 = unet(
                                torch.cat([sample_1["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_1["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_1,
                            ).sample
                            noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                            noise_pred_1 = noise_pred_uncond_1 + config.sample.guidance_scale * (
                                noise_pred_text_1 - noise_pred_uncond_1
                            )
                            
                            noise_ref_pred_1 = ref(
                                torch.cat([sample_1["latents"][train_b_start: train_b_end]] * 2),
                                torch.cat([sample_1["timesteps"][train_b_start: train_b_end]] * 2),
                                embeds_1,
                            ).sample
                            noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                            noise_ref_pred_1 = noise_ref_pred_uncond_1 + config.sample.guidance_scale * (
                                noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                            )
                        else:
                            noise_pred_0 = unet(
                                sample_0["latents"][train_b_start: train_b_end], 
                                sample_0["timesteps"][train_b_start: train_b_end], 
                                embeds_0,
                            ).sample
                            noise_ref_pred_0 = ref(
                                sample_0["latents"][train_b_start: train_b_end], 
                                sample_0["timesteps"][train_b_start: train_b_end], 
                                embeds_0,
                            ).sample
                            
                            noise_pred_1 = unet(
                                sample_1["latents"][train_b_start: train_b_end], 
                                sample_1["timesteps"][train_b_start: train_b_end], 
                                embeds_1,
                            ).sample
                            noise_ref_pred_1 = ref(
                                sample_1["latents"][train_b_start: train_b_end], 
                                sample_1["timesteps"][train_b_start: train_b_end], 
                                embeds_1,
                            ).sample
                    
                    
                    total_prob_0 = step_with_logprob(
                        pipeline.scheduler,
                        noise_pred_0,
                        sample_0["timesteps"][train_b_start: train_b_end],
                        sample_0["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_0["next_latents"][train_b_start: train_b_end],
                    )
                    total_ref_prob_0 = step_with_logprob(
                        pipeline.scheduler,
                        noise_ref_pred_0,
                        sample_0["timesteps"][train_b_start: train_b_end],
                        sample_0["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_0["next_latents"][train_b_start: train_b_end],
                    )
                    total_prob_1 = step_with_logprob(
                        pipeline.scheduler,
                        noise_pred_1,
                        sample_1["timesteps"][train_b_start: train_b_end],
                        sample_1["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_1["next_latents"][train_b_start: train_b_end],
                    )
                    total_ref_prob_1 = step_with_logprob(
                        pipeline.scheduler,
                        noise_ref_pred_1,
                        sample_1["timesteps"][train_b_start: train_b_end],
                        sample_1["latents"][train_b_start: train_b_end],
                        eta=config.sample.eta,
                        prev_sample=sample_1["next_latents"][train_b_start: train_b_end],
                    )
                
                    ratio_0 = torch.clamp(torch.exp(total_prob_0-total_ref_prob_0),1 - config.train.eps, 1 + config.train.eps)
                    ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1),1 - config.train.eps, 1 + config.train.eps)
                    loss = -torch.log(torch.sigmoid(config.train.beta*(torch.log(ratio_0)) - config.train.beta*(torch.log(ratio_1)))).mean()
                    
                    avg_loss = accelerator.reduce(loss.detach(), reduction='mean')
                    train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps
                    
                    # batch size              
                    win_ratio_sum =  accelerator.reduce(ratio_0.detach(), reduction='sum')       
                    lose_ratio_sum =  accelerator.reduce(ratio_1.detach(), reduction='sum')       
                    
                    avg_win_ratio = (win_ratio_sum.sum() / (win_ratio_sum.shape[0] * accelerator.num_processes)).item()
                    avg_lose_ratio = (lose_ratio_sum.sum() / (lose_ratio_sum.shape[0] * accelerator.num_processes)).item()

                    train_ratio_win += avg_win_ratio / accelerator.gradient_accumulation_steps
                    train_ratio_lose += avg_lose_ratio / accelerator.gradient_accumulation_steps

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_para, config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    # log training-related stuff
                    info = {
                        "epoch": epoch, 
                        "global_step": global_step, 
                        "train_loss": train_loss,
                        "train_ratio_win": train_ratio_win,
                        "train_ratio_lose": train_ratio_lose,
                        "lr": optimizer.param_groups[0]['lr'],
                    }
                    accelerator.log(info, step=global_step)
                    global_step += 1
                    train_loss = 0.0
                    train_ratio_win = 0.0
                    train_ratio_lose = 0.0

            if (
                dataset_batch_idx == len(data_loader) - 1 and 
                accelerator.gradient_state.in_dataloader
            ):
                accelerator.gradient_state.active_dataloader.end_of_dataloader = True

        ########## save ckpt and evaluation ##########
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_interval == 0:
                accelerator.save_state(os.path.join(config.logdir, config.run_name, f'checkpoint_{epoch}'))
                with open(os.path.join(config.logdir, config.run_name, f'checkpoint_{epoch}', 'global_step.json'), 'w') as f:
                    json.dump({'global_step': global_step}, f)
            if  (epoch + 1) % config.eval_interval == 0 and config.validation_prompts is not None:
                prompt_info = f"Running validation... \n Generating {config.num_validation_audios} audios with prompt:\n"
                for prompt in config.validation_prompts:
                    prompt_info = prompt_info + prompt + '\n'

                logger.info(prompt_info)
                # create pipeline
                unet.eval()
                pipeline.unet.eval()
                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None

                audio_logs = []
                for idx, validation_prompt in enumerate(config.validation_prompts):
                    with torch.cuda.amp.autocast():
                        audios = [
                            pipeline(
                                prompt=validation_prompt,
                                num_inference_steps=config.sample.num_steps,
                                generator=generator,
                                guidance_scale=config.sample.guidance_scale,
                            ).audios[0]
                            for _ in range(config.num_validation_audios)
                        ]
                    audio_logs.append(
                        {
                            "audios": audios, 
                            "prompts": validation_prompt,
                        }
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        formatted_audios = []
                        for log in audio_logs:
                            audios = log["audios"]
                            validation_prompt = log["prompts"]
                            for idx, audio in enumerate(audios):
                                audio = wandb.Audio(audio, caption=validation_prompt)
                                formatted_audios.append(audio)
                        tracker.log({"validation": formatted_audios})
                unet.train()
                pipeline.unet.train()
                torch.cuda.empty_cache()
    
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        GradTTSPipeline.save_lora_weights(
            save_directory=os.path.join(config.logdir, config.run_name),
            unet_lora_layers=unet_lora_state_dict,
        )
    
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
