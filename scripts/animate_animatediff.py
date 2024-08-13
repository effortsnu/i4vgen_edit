import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision
from torchvision.utils import save_image

from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from i4vgen.animatediff.models.unet import UNet3DConditionModel
from i4vgen.animatediff.pipelines.pipeline_animation import AnimationPipeline
from i4vgen.animatediff.utils.util import save_videos_grid
from i4vgen.animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange
from pathlib import Path

import ImageReward as RM
from PIL import Image
import torchvision.transforms as transforms
from i4vgen.animatediff.models.sparse_controlnet import SparseControlNetModel
import numpy as np


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir, exist_ok=True)

    config  = OmegaConf.load(args.config)

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

            ### >>> create validation pipeline >>> ###
            tokenizer           = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder        = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
            vae                 = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()    
            unet                = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()
            image_reward_model  = RM.load("ImageReward-v1")
        
        image_path = None

            # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_image", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_path = model_config.controlnet_image
            if isinstance(image_path, str): image_path = [image_path]

            print(f"controlnet image path:")
            for path in image_path: print(path)
            assert len(image_path) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_path]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
            

            if is_xformers_available(): 
                unet.enable_xformers_memory_efficient_attention()
                if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
            else: 
                assert False

            pipeline = AnimationPipeline(
                image_reward_model, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                controlnet=controlnet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            pipeline = load_weights(
                pipeline,
                # motion module
                motion_module_path         = motion_module,
                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                # image layers
                dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                lora_model_path            = model_config.get("lora_model_path", ""),
                lora_alpha                 = model_config.get("lora_alpha", 0.8),
                # Support PIA
                PIA_path                   = model_config.get("PIA_path", ""),
            ).to("cuda")

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"Processing the ({prompt}) prompt")

                # Save intermediate results 
                assert image_path is not None
                # There is only one control image
                sample = pipeline(
                    image_path[0],
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                    use_fp16            = model_config.use_fp16,

                    controlnet_images = controlnet_images,
                    controlnet_image_index = model_config.get("controlnet_image_index", [0]),
                    controlnet_use_simplified_condition_embedding = controlnet.use_simplified_condition_embedding,
                )

                sample, candidate_images, ni_vsds_video = sample.videos, sample.candidate_images, sample.ni_vsds_video
                
                video_name = f"{savedir}/{prompt}-{random_seed}.mp4"
                sample_mp4 = sample[0].contiguous().permute(1, 2, 3, 0).contiguous()
                sample_mp4 = (sample_mp4 * 255)
                sample_mp4 = sample_mp4.to(dtype=torch.uint8)
                torchvision.io.write_video(video_name, sample_mp4, fps=8)

                '''candidate_images'''
                sample_png = rearrange(candidate_images, "b c f h w -> b c h (f w)").contiguous()
                png_name = f"{savedir}/{prompt}-{random_seed}-candidate-images.png"
                save_image(sample_png, png_name)

                '''ni_vsds_video'''
                # video_name = f"{savedir}/{prompt}-{random_seed}-ni-vsds-video.mp4"
                # sample_mp4 = ni_vsds_video[0].contiguous().permute(1, 2, 3, 0).contiguous()
                # sample_mp4 = (sample_mp4 * 255)
                # sample_mp4 = sample_mp4.to(dtype=torch.uint8)
                # torchvision.io.write_video(video_name, sample_mp4, fps=8)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/home/jhpark/AnimateDiff/models/StableDiffusion",)
    parser.add_argument("--inference_config",      type=str, default="configs/animatediff_configs/inference/inference-v2.yaml")    
    parser.add_argument("--config",                type=str, required=True)

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
