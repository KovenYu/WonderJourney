import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
from copy import deepcopy
import json

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import sys
sys.path.append('midas_module')
from midas_module.midas.model_loader import load_model
import torch.nn.functional as F

from models.models import KeyframeGen, KeyframeInterp, save_point_cloud_as_ply
from util.finetune_utils import finetune_depth_model, finetune_decoder
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler
from util.utils import load_example_yaml, merge_frames, merge_keyframes
from util.segment_utils import create_mask_generator


def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)

    video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch, vmax=None):
    rendered_depth = model.rendered_depths[epoch].clamp(0).cpu().numpy()
    depth = model.depths[epoch].clamp(0).cpu().numpy()
    save_root = Path(model.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "inpaint_input_image").mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)
    (save_root / "depth").mkdir(exist_ok=True, parents=True)

    model.inpaint_input_image[epoch].save(save_root / "inpaint_input_image" / f"{epoch}.png")
    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.post_masks[epoch][0]).save(save_root / "post_masks" / f"{epoch}.png")
    ToPILImage()(model.rendered_images[epoch][0]).save(save_root / "rendered_images" / f"{epoch}.png")
    save_depth_map(rendered_depth, save_root / "rendered_depths" / f"{epoch}.png", vmax=vmax)
    save_depth_map(depth, save_root / "depth" / f"{epoch}.png", vmax=vmax, save_clean=True)

    if hasattr(model, "outter_masks"):
        (save_root / "outter_masks").mkdir(exist_ok=True, parents=True)
        ToPILImage()(model.outter_masks[epoch]).save(save_root / "outter_masks" / f"{epoch}.png")
    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def run(config):

    ###### ------------------ Load modules ------------------ ######

    if config['skip_gen']:
        kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
    else:
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        kfgen_save_folder = Path(config['runs_dir']) / f"{dt_string}_kfgen"
    kfgen_save_folder.mkdir(exist_ok=True, parents=True)
    cutoff_depth = config['fg_depth_range'] + config['depth_shift']
    vmax = cutoff_depth * 2
    inpainting_resolution_gen = config['inpainting_resolution_gen']
    seeding(config["seed"])

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    mask_generator = create_mask_generator()

    all_rundir = []
    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "
    all_keyframes = [start_keyframe]
    
    if isinstance(control_text, list):
        config['num_scenes'] = len(control_text)
    pt_gen = TextpromptGen(config['runs_dir'], isinstance(control_text, list))
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = style_prompt + ', ' + content_prompt

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16",
        ).to(config["device"])
    inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.scheduler = prepare_scheduler(inpainter_pipeline.scheduler)
    vae = AutoencoderKL.from_pretrained(config["stable_diffusion_checkpoint"], subfolder="vae").to(config["device"])

    rotation_path = config['rotation_path']
    assert len(rotation_path) >= config['num_scenes'] * config['num_keyframes']

    ###### ------------------ Main loop ------------------ ######

    for i in range(config['num_scenes']):
        if config['use_gpt']:
            control_text_this = control_text[i] if isinstance(control_text, list) else None
            scene_dict = pt_gen.run_conversation(scene_name=scene_dict['scene_name'], entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], control_text=control_text_this)
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        
        for j in range(config['num_keyframes']):

            ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######

            if config['skip_gen']:
                kf_gen_dict = torch.load(kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")
                kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
                kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                kf2_mask = kf_gen_dict['kf2_mask']
                inpainting_prompt, adaptive_negative_prompt = kf_gen_dict['inpainting_prompt'], kf_gen_dict['adaptive_negative_prompt']
                rotation = kf_gen_dict['rotation']
            else:
                rotation = rotation_path[i*config['num_keyframes'] + j]
                regen_negative_prompt = ""
                config['inpainting_resolution_gen'] = inpainting_resolution_gen
                for regen_id in range(config['regenerate_times'] + 1):
                    if regen_id > 0:
                        seeding(-1)
                    depth_model, _, _, _ = load_model(torch.device("cuda"), 'dpt_beit_large_512.pt', 'dpt_beit_large_512', optimize=False)
                    # first keyframe is loaded and estimated depth
                    kf_gen = KeyframeGen(config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                                        start_keyframe, inpainting_prompt, regen_negative_prompt + adaptive_negative_prompt,
                                        segment_model=segment_model, segment_processor=segment_processor).to(config["device"])
                    save_root = Path(kf_gen.run_dir) / "images"
                    kf_idx = 0

                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_original', vmin=0, vmax=vmax)
                    kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth)
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_processed', vmin=0, vmax=vmax)
                    evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

                    kf_idx = 1
                    render_output = kf_gen.render(kf_idx)
                    inpaint_output = kf_gen.inpaint(render_output["rendered_image"], render_output["inpaint_mask"])

                    regenerate_information = {}
                    if config['enable_regenerate'] and regen_id <= config['regenerate_times'] -1:
                        gpt_border, gpt_blur = pt_gen.evaluate_image(ToPILImage()(inpaint_output['inpainted_image'][0]), eval_blur=False)
                        regenerate_information['gpt_border'] = gpt_border
                        regenerate_information['gpt_blur'] = gpt_blur
                        if gpt_border:
                            print("chatGPT-4 says the image has border!")
                            regen_negative_prompt += "border, "
                        if gpt_blur:
                            print("chatGPT-4 says the image has blurry effect!")
                            regen_negative_prompt += "blur, "
                        regenerate = gpt_border
                    else:
                        regenerate = False

                    with open(save_root / 'regenerate_info.json', 'w') as json_file:
                        json.dump(regenerate_information, json_file, indent=4)
                    
                    if not regenerate:
                        break
                    if regen_id == config['regenerate_times'] -1:
                        print("Regenerating faild after {} times".format(config['regenerate_times']))
                        if gpt_border:
                            print("Use crop to solve border problem!")
                            config['inpainting_resolution_gen'] = 560
                        else:
                            break

                    # get memory back
                    depth_model = kf_gen.depth_model.to('cpu')
                    kf_gen.depth_model = None
                    del depth_model
                    empty_cache()

                if config["finetune_decoder_gen"]:
                    ToPILImage()(inpaint_output["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
                    finetune_decoder(config, kf_gen, render_output, inpaint_output, config['num_finetune_decoder_steps'])

                kf_gen.update_images_and_masks(inpaint_output["latent"], render_output["inpaint_mask"])

                kf2_depth_should_be = render_output['rendered_depth']
                mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be < cutoff_depth + kf_gen.kf_delta_t)
                mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be >= cutoff_depth + kf_gen.kf_delta_t)

                # with torch.no_grad():
                #     kf2_before_ft_depth, _ = kf_gen.get_depth(kf_gen.images[kf_idx])  # pix depth under kf2 frame
                if config["finetune_depth_model"]:
                    finetune_depth_model(config, kf_gen, kf2_depth_should_be, kf_idx, mask_align=mask_to_align_depth, 
                                        mask_cutoff=mask_to_cutoff_depth, cutoff_depth=cutoff_depth + kf_gen.kf_delta_t)
                with torch.no_grad():
                    kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
                    kf_gen.depths.append(kf2_ft_depth_original), kf_gen.disparities.append(kf2_ft_disp_original)
                # save_depth_map(kf2_before_ft_depth.detach().cpu().numpy(), save_root / 'kf2_before_ft_depth', vmin=0, vmax=vmax)
                # save_depth_map(kf2_depth_should_be_processed.detach().cpu().numpy(), save_root / 'kf2_depth_should_be_processed', vmin=0, vmax=vmax)
                # save_depth_map(kf2_depth_should_be_original.detach().cpu().numpy(), save_root / 'kf2_depth_should_be_original', vmin=0, vmax=vmax)
                # save_depth_map(kf2_ft_depth_original.cpu().numpy(), save_root / 'kf2_ft_depth_original', vmin=0, vmax=vmax)

                # get memory back
                depth_model = kf_gen.depth_model.to('cpu')
                kf_gen.depth_model = None
                del depth_model
                empty_cache()

                kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth + kf_gen.kf_delta_t)
                save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed', vmin=0, vmax=vmax)
                    
                kf_gen.vae.decoder = deepcopy(kf_gen.decoder_copy)
                evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

                start_keyframe = ToPILImage()(kf_gen.images[1][0])
                all_keyframes.append(start_keyframe)

                kf1_depth, kf2_depth = kf_gen.depths[0], kf_gen.depths[-1]
                kf1_image, kf2_image = kf_gen.images[0], kf_gen.images[1]
                kf1_camera, kf2_camera = kf_gen.cameras[0], kf_gen.cameras[1]
                kf2_mask = render_output["inpaint_mask_512"]
                kf_gen_dict = {'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth, 'kf1_image': kf1_image, 'kf2_image': kf2_image, 
                            'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera, 'kf2_mask': kf2_mask, 'inpainting_prompt': inpainting_prompt, 
                            'adaptive_negative_prompt': adaptive_negative_prompt, 'rotation': rotation}
                torch.save(kf_gen_dict, kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")

                if config['skip_interp']:
                    kf_gen = kf_gen.to('cpu')
                    del kf_gen
                    empty_cache()
                    continue

            ###### ------------------ Keyframe interpolation (completing point clouds and rendering) ------------------ ######
                
            is_last_scene = i == config['num_scenes'] - 1
            is_last_keyframe = j == config['num_keyframes'] - 1
            try:
                is_next_rotation = rotation_path[i*config['num_keyframes'] + j + 1] != 0
            except IndexError:
                is_next_rotation = False
            try:
                is_previous_rotation = rotation_path[i*config['num_keyframes'] + j - 1] != 0
            except IndexError:
                is_previous_rotation = False
            is_beginning = i == 0 and j == 0
            speed_up = (rotation == 0) and ((is_last_scene and is_last_keyframe) or is_next_rotation)
            speed_down = (rotation == 0) and (is_beginning or is_previous_rotation)
            total_frames = config["frames"]
            total_frames = total_frames + config["frames"] // 5 if speed_up else total_frames
            total_frames = total_frames + config["frames"] // 5 if speed_down else total_frames
            kf_interp = KeyframeInterp(config, inpainter_pipeline, None, vae, rotation, 
                                   ToPILImage()(kf1_image[0]), inpainting_prompt, adaptive_negative_prompt,
                                   kf2_upsample_coef=config['kf2_upsample_coef'], kf1_image=kf1_image, kf2_image=kf2_image,
                                   kf1_depth=kf1_depth, kf2_depth=kf2_depth, kf1_camera=kf1_camera, kf2_camera=kf2_camera, kf2_mask=kf2_mask,
                                   speed_up=speed_up, speed_down=speed_down, total_frames=total_frames
                                   ).to(config["device"])
            save_root = Path(kf_interp.run_dir) / "images"
            save_root.mkdir(exist_ok=True, parents=True)
            ToPILImage()(kf1_image[0]).save(save_root / "kf1.png")
            ToPILImage()(kf2_image[0]).save(save_root / "kf2.png")

            kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_image_upsample = kf_interp.upsample_kf2()

            kf_interp.update_additional_point_cloud(kf2_depth_upsample, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
            inconsistent_additional_point_index = kf_interp.visibility_check()
            kf2_depth_updated = kf_interp.update_additional_point_depth(inconsistent_additional_point_index, depth=kf2_depth_upsample, mask=kf2_mask_upsample)
            # save_depth_map(kf2_depth_updated.detach().cpu().numpy(), save_root / 'kf2_depth_updated', vmin=0, vmax=vmax)
            kf_interp.reset_additional_point_cloud()
            kf_interp.update_additional_point_cloud(kf2_depth_updated, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
            
            kf_interp.depths[0] = F.interpolate(kf2_depth_updated, size=(512, 512), mode="nearest")
            # save_depth_map(kf_interp.depths[0].detach().cpu().numpy(), save_root / 'kf2_depth.png', vmin=0, vmax=cutoff_depth*0.95, save_clean=True)
            # save_point_cloud_as_ply(kf_interp.additional_points_3d*500, kf_interp.run_dir / 'kf2_point_cloud.ply', kf_interp.additional_colors)
            # save_point_cloud_as_ply(kf_interp.points_3d *500, kf_interp.run_dir / 'kf1_point_cloud.ply', kf_interp.kf1_colors)
            evaluate_epoch(kf_interp, 0, vmax=vmax)

            for epoch in tqdm(range(1, total_frames + 1)):
                render_output_kf1 = kf_interp.render_kf1(epoch)

                inpaint_output = kf_interp.inpaint(render_output_kf1["rendered_image"], render_output_kf1["inpaint_mask"])

                if config["finetune_decoder_interp"]:
                    finetune_decoder(config, kf_interp, render_output_kf1, inpaint_output, config["num_finetune_decoder_steps_interp"])

                # use latent to get fine-tuned image; center crop if needed; then update image/mask/depth
                kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"])

                kf_interp.update_additional_point_cloud(render_output_kf1["rendered_depth"], kf_interp.images[-1], append_depth=True)

                # reload decoder
                kf_interp.vae.decoder = deepcopy(kf_interp.decoder_copy)
                with torch.no_grad():
                    kf_interp.images_orig_decoder.append(kf_interp.decode_latents(inpaint_output["latent"]).detach())
                evaluate_epoch(kf_interp, epoch, vmax=cutoff_depth*0.95)
                empty_cache()

            kf_interp.images.append(kf1_image)  # so that the last frame is KF1
            evaluate(kf_interp)
            # save_point_cloud_as_ply(torch.cat([kf_interp.points_3d, kf_interp.additional_points_3d], dim=0)*500, kf_interp.run_dir / 'final_point_cloud.ply', torch.cat([kf_interp.kf1_colors, kf_interp.additional_colors], dim=0))

            all_rundir.append(kf_interp.run_dir)

    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    save_dir = Path(config['runs_dir']) / f"{dt_string}_merged"
    if not config['skip_interp']:
        merge_frames(all_rundir, save_dir=save_dir, fps=config["save_fps"], is_forward=True, save_depth=False, save_gif=False)
    merge_keyframes(all_keyframes, save_dir=save_dir)
    pt_gen.write_all_content(save_dir=save_dir)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config"
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run(config)