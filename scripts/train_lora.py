import os
import sys
import lpips
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from typing import Tuple
from einops import rearrange
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import torchvision.utils as vutils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.datasets import NoiseDataset
from model.loss import gram_loss
from model.pipeline_difix import DifixPipeline

def load_ckpt_from_state_dict(net_difix, optimizer, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cpu")
    
    if "state_dict_vae" in sd:
        _sd_vae = net_difix.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net_difix.vae.load_state_dict(_sd_vae)
    _sd_unet = net_difix.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net_difix.unet.load_state_dict(_sd_unet)
        
    optimizer.load_state_dict(sd["optimizer"])
    
    return net_difix, optimizer


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    net_difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    net_difix.to("cuda")
    net_difix.set_train(lora_rank_vae = args.lora_rank_vae)
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    net_vgg = torchvision.models.vgg16(pretrained=True).features

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    layers_to_opt = []
    layers_to_opt += list(net_difix.unet.parameters())
    
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = (
        layers_to_opt
        + list(net_difix.vae.decoder.skip_conv_1.parameters())
        + list(net_difix.vae.decoder.skip_conv_2.parameters())
        + list(net_difix.vae.decoder.skip_conv_3.parameters())
        + list(net_difix.vae.decoder.skip_conv_4.parameters())
    )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    phase1_output_dir = args.output_dir.replace("/phase2", "/phase1")
    dataset_train = NoiseDataset(
        dataset_path=[args.dataset_path, phase1_output_dir],
        split="train",
        image_processor=net_difix.image_processor,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_image_dir = os.path.join(phase1_output_dir, "Lora", "val", "images")
    if os.path.isdir(val_image_dir) and len(os.listdir(val_image_dir)) > 0:
        dataset_val = NoiseDataset(
            dataset_path=[args.dataset_path, phase1_output_dir],
            split="test",
            image_processor=net_difix.image_processor,
        )
        dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    else:
        dl_val = None
        print("No LoRA validation renders found; skip validation visualization.")

    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(
                ckpt_files,
                key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")),
            )
            print("=" * 50)
            print(f"Loading checkpoint from {ckpt_files[-1]}")
            print("=" * 50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("=" * 50)
            print(f"Loading checkpoint from {args.resume}")
            print("=" * 50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_difix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare everything with our `accelerator`.
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    #
    os.makedirs(args.output_dir,exist_ok=True)
    
    num_training_epochs = int(args.max_train_steps/len(dataset_train)) + 1
    # start the training loop
    for epoch in range(0, num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                global_step += 1
                x_src = batch["input_img"].to(accelerator.device)
                x_tgt = batch["gt_img"].to(accelerator.device)
                prompt = "remove degradation"
                B, V, C, H, W = x_src.shape
                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_src = rearrange(x_src, 'b v c h w -> (b v) c h w')
                x_tgt_pred = net_difix(
                    prompt,
                    image=x_src,
                    num_inference_steps=1,
                    timesteps=[199],
                    guidance_scale=0.0,
                )

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips 
                
                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 256, 256
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = (
                            gram_loss(
                                x_tgt_pred_renorm.to(weight_dtype),
                                x_tgt_renorm.to(weight_dtype),
                                net_vgg,
                            )
                            * args.lambda_gram
                        )
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)
                        # viz some images
                    if global_step % args.viz_freq == 0:
                        vis_path = os.path.join(args.output_dir, "vis")
                        os.makedirs(vis_path,exist_ok=True)
                        concatenated_tensor = []
                        for i in range(B):
                            concatenated_tensor.append(x_src[i].detach().clone())
                            concatenated_tensor.append(x_tgt_pred[i].detach().clone())
                            concatenated_tensor.append(x_tgt[i].detach().clone())
                        concatenated_tensor = torch.stack(concatenated_tensor,dim=0)
                        vutils.save_image((concatenated_tensor+1.0)/2.0, f"{vis_path}/{global_step}.png")
                            # checkpoint the model
                    if global_step % args.save_step == 0:
                        save_path = os.path.join(args.output_dir,f"model/{global_step}/vae")
                        if not os.path.exists(save_path):
                            
                            os.makedirs(save_path,exist_ok=True)
                        accelerator.save_model(net_difix.vae, save_path)
                        save_path = os.path.join(args.output_dir,f"model/{global_step}/unet")
                        if not os.path.exists(save_path):
                            
                            os.makedirs(save_path,exist_ok=True)
                        accelerator.save_model(net_difix.unet, save_path)
                    if dl_val is not None and global_step % args.eval_freq == 0:
                        with torch.no_grad():
                            for step, batch in enumerate(dl_val):
                                x_src = batch["input_img"].to(accelerator.device)
                                prompt = "remove degradation"
                                B, V, C, H, W = x_src.shape
                                x_src = rearrange(x_src, 'b v c h w -> (b v) c h w')
                                x_tgt_pred = net_difix(
                                    prompt,
                                    image=x_src,
                                    num_inference_steps=1,
                                    timesteps=[199],
                                    guidance_scale=0.0,
                                )
                                vis_path = os.path.join(args.output_dir, f"val/{global_step}")
                                os.makedirs(vis_path,exist_ok=True)
                                concatenated_tensor = torch.cat(
                                    [x_src.detach().clone(), x_tgt_pred.detach().clone()]
                                )
                                image_grid = vutils.make_grid(
                                    concatenated_tensor,
                                    nrow=2,
                                    padding=2,
                                    normalize=False,
                                )
                                vutils.save_image((image_grid+1.0)/2.0, f"{vis_path}/{step}.png")
                    
                               
            torch.cuda.empty_cache()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=1000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--save_step", default=1000, type=int)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", '
            '"cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. "
            "For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--data_factor", default=8, type=int)
    parser.add_argument("--crop_size", default=[],type= Tuple[int, int])

    args = parser.parse_args()

    main(args)
