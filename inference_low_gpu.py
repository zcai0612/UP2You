import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.utils import save_image
from up2you.utils.img_utils import load_image, process_image_rgba
from up2you.models.feature_aggregator import FeatureAggregator
from up2you.models.encoder.dinov2_wrapper import Dinov2Wrapper
from up2you.models.shape_predictor import ShapePredictor
from up2you.pipelines.pipeline_mvpuzzle_i2mv_sd21 import UP2YouI2MVSDPipeline
from up2you.pipelines.pipeline_mvpuzzle_mv2normal_sd21 import UP2YouMV2NormalPipeline
from up2you.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from up2you.utils.weight_map_utils import weight_map_to_heatmap
from up2you.utils.smpl_utils.apose_renderer import AposeRenderer
from up2you.utils.mesh_utils.reconstructor import Reconstructor
from diffusers import DDPMScheduler
from einops import rearrange
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import random
from up2you.utils.mesh_utils.mesh_util import save_obj_mesh
import shutil
from up2you.utils.mesh_utils.mesh_common_renderer import CommonRenderer
from up2you.utils.video_utils import tensor_to_video
import gc


def manual_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def segment_rgbs(
    rgbs,
    seg_model,
    device,
):
    processed_rgbs = []
    for image in rgbs:
        if image.mode != 'RGBA':
            input_image = transform_image(image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = seg_model(input_image)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(image.size)
                image.putalpha(mask)
        processed_rgbs.append(image)
    return processed_rgbs

def preprocess_ref_imgs(
    data_dir,
    seg_model,
    device,
    output_ref_imgs_dir=None,
):
    ref_img_names = os.listdir(data_dir)
    ref_img_names.sort()
    ref_pils = []
    for ref_img_name in ref_img_names:
        ref_img_path = os.path.join(data_dir, ref_img_name)
        ref_pil = Image.open(ref_img_path).convert("RGB")
        ref_pil = ImageOps.exif_transpose(ref_pil)
        ref_pils.append(ref_pil)
    ref_rgbas = segment_rgbs(ref_pils, seg_model, device)
    ref_imgs = [process_image_rgba(ref_rgba, ratio=1.85/2) for ref_rgba in ref_rgbas]
    if output_ref_imgs_dir is not None:
        for idx, ref_img in enumerate(ref_imgs):
            ref_img.save(os.path.join(output_ref_imgs_dir, f"{idx}.png"))

    ref_img_tensor_list = []
    ref_alpha_tensor_list = []
    for ref_img in ref_imgs:
        ref_img_tensor, ref_alpha_tensor = load_image(ref_img, 768, 768, return_alpha=True)
        ref_img_tensor_list.append(ref_img_tensor)
        ref_alpha_tensor_list.append(ref_alpha_tensor)
    ref_img_tensor = torch.stack(ref_img_tensor_list)
    ref_alpha_tensor = torch.stack(ref_alpha_tensor_list)
    
    return ref_img_tensor.permute(0, 3, 1, 2).to(device), ref_alpha_tensor.to(device)

def stage1_feature_extraction(
    ref_rgbs,
    ref_alphas,
    shape_predictor_path,
    device,
    dtype,
):
    image_encoder = Dinov2Wrapper(
        device=device,
        model_name="dinov2_vitl14",
        image_size=518,
    )
    image_encoder.to(device)

    shape_predictor = ShapePredictor(
        embed_dim=1024,
        depth=2,
        trunk_depth=2
    )
    shape_predictor.load_state_dict(torch.load(shape_predictor_path, map_location="cpu"))
    shape_predictor.to(device)
    shape_predictor.eval()

    with torch.no_grad():
        ref_img_feats = image_encoder(ref_rgbs)
        ref_img_feats = rearrange(
            ref_img_feats,
            "(B Nr) H W C -> B Nr H W C",
            B=1
        )
        betas = shape_predictor(ref_img_feats)

    del image_encoder
    del shape_predictor
    clear_gpu_memory()

    return ref_img_feats, betas

def stage2_pose_generation(
    betas,
    device,
):
    apose_renderer = AposeRenderer(
        device=device,
    )
    
    target_poses, _, smplx_v, smplx_f = apose_renderer(betas.reshape(1, 10), height=768, width=768, return_mesh=True)
    
    del apose_renderer
    clear_gpu_memory()
    
    return target_poses, smplx_v, smplx_f

def stage3_weight_map_generation(
    target_poses,
    ref_img_feats,
    ref_alphas,
    feature_aggregator_path,
    device,
):
    feature_aggregator = FeatureAggregator(
        pose_img_size=512,
        pose_img_in_chans=3,
        pose_patch_embed_type="pose_encoder",
        embed_dim=1024,
        patch_size=16,
        depth=1,
        aa_order=['self', 'self', 'cross', 'cross'],
        weight_norm="none",
        use_mask=True,
        smooth_method="avgpool",
        kernel_size=3,
    )

    feature_aggregator.load_state_dict(torch.load(feature_aggregator_path, map_location="cpu"))
    feature_aggregator.to(device)
    feature_aggregator.eval()

    target_pose_imgs = rearrange(
        target_poses,
        "(B Nv) C H W -> B Nv H W C",
        Nv=6
    )
    ref_alphas = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", B=1)
    
    with torch.no_grad():
        weight_maps = feature_aggregator(
            target_pose_imgs=target_pose_imgs,
            ref_img_feats=ref_img_feats,
            ref_alphas=ref_alphas,
        )

    del feature_aggregator
    clear_gpu_memory()

    return weight_maps

def stage4_rgb_generation(
    target_poses,
    ref_rgbs,
    weight_maps,
    base_model_path,
    rgb_adapter_path,
    device,
    dtype,
):
    rgb_pipe = UP2YouI2MVSDPipeline.from_pretrained(
        base_model_path,
    )

    rgb_pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        rgb_pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )

    rgb_pipe.init_custom_adapter(
        num_views=6,
        mode='topk',
    )
    rgb_pipe.load_custom_adapter(
        rgb_adapter_path, weight_name='custom_adapter.safetensors'
    )

    rgb_pipe.to(device=device, dtype=dtype)
    rgb_pipe.cond_encoder.to(device=device, dtype=dtype)
    rgb_pipe.enable_vae_slicing()

    images = rgb_pipe(
        prompt=["Multi-view Human, Full Body, High Quality, HDR"],
        control_image=target_poses,
        num_images_per_prompt=6,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50,
        guidance_scale=3.0,
        height=768,
        width=768,
        reference_rgbs=ref_rgbs,
        weight_maps=weight_maps,
    ).images

    del rgb_pipe
    clear_gpu_memory()

    return images

def stage5_normal_generation(
    target_poses,
    mv_rgbs,
    base_model_path,
    normal_adapter_path,
    device,
    dtype,
):
    normal_pipe = UP2YouMV2NormalPipeline.from_pretrained(
        base_model_path,
    )

    normal_pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        normal_pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )

    normal_pipe.init_custom_adapter(
        num_views=6,
    )
    normal_pipe.load_custom_adapter(
        normal_adapter_path, weight_name='custom_adapter.safetensors'
    )

    normal_pipe.to(device=device, dtype=dtype)
    normal_pipe.cond_encoder.to(device=device, dtype=dtype)
    normal_pipe.enable_vae_slicing()

    normals = normal_pipe(
        prompt=["Multi-view Human, Full Body, Normal Map, High Quality, HDR"],
        control_image=target_poses,
        num_images_per_prompt=6,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50,
        guidance_scale=3.0,
        height=768,
        width=768,
        reference_rgbs=mv_rgbs,
    ).images

    del normal_pipe
    clear_gpu_memory()

    return normals

def stage6_reconstruction(
    images_rgba,
    normals_rgba,
    smplx_mesh_path,
    pred_meshes_output_dir,
    device,
):
    reconstructor = Reconstructor(
        device=device,
    )

    gen_obj_path = reconstructor.run(
        color_pils=images_rgba,
        normal_pils=normals_rgba,
        smplx_obj_path=smplx_mesh_path,
        output_dir=pred_meshes_output_dir,
        replace_hand=True,
    )

    del reconstructor
    clear_gpu_memory()

    return gen_obj_path

def stage7_video_rendering(
    gen_obj_path,
    output_dir,
):
    video_renderer = CommonRenderer(
        resolution=1024, return_rgba=True
    )

    render_rgbs, render_normals = video_renderer.render_video(gen_obj_path, None, normal_type="camera", background_color="white")
    tensor_to_video(render_rgbs, os.path.join(output_dir, "render.mp4"))
    tensor_to_video(render_normals, os.path.join(output_dir, "normal.mp4"))

    del video_renderer
    clear_gpu_memory()

def infer_in_the_wild_low_gpu(
    ref_img_dir,
    base_model_path,
    rgb_adapter_path,
    normal_adapter_path,
    feature_aggregator_path,
    shape_predictor_path,
    segment_model_name,
    device,
    dtype,
    output_dir,
):
    pred_rgbs_output_dir = os.path.join(output_dir, "pred_rgbs")
    pred_corr_maps_output_dir = os.path.join(output_dir, "pred_corr_maps")
    pred_normals_output_dir = os.path.join(output_dir, "pred_normals")
    pred_meshes_output_dir = os.path.join(output_dir, "meshes")
    output_ref_imgs_dir = os.path.join(output_dir, "ref_imgs")

    os.makedirs(pred_rgbs_output_dir, exist_ok=True)
    os.makedirs(pred_corr_maps_output_dir, exist_ok=True)
    os.makedirs(pred_normals_output_dir, exist_ok=True)
    os.makedirs(pred_meshes_output_dir, exist_ok=True)
    os.makedirs(output_ref_imgs_dir, exist_ok=True)

    seg_model = AutoModelForImageSegmentation.from_pretrained(segment_model_name, trust_remote_code=True)
    seg_model.eval()
    seg_model.to(device)

    ref_rgbs, ref_alphas = preprocess_ref_imgs(ref_img_dir, seg_model, device, output_ref_imgs_dir)
    
    del seg_model
    clear_gpu_memory()

    print("Stage 1: Feature Extraction")
    ref_img_feats, betas = stage1_feature_extraction(
        ref_rgbs, ref_alphas, shape_predictor_path, device, dtype
    )

    print("Stage 2: Pose Generation")
    target_poses, smplx_v, smplx_f = stage2_pose_generation(betas, device)
    
    save_image(ref_rgbs, os.path.join(output_dir, "ref_rgbs.png"))
    save_image(target_poses, os.path.join(output_dir, "target_pose.png"))
    smplx_mesh_path = os.path.join(pred_meshes_output_dir, "smplx_mesh.obj")
    save_obj_mesh(smplx_mesh_path, smplx_v, smplx_f)

    print("Stage 3: Weight Map Generation")
    weight_maps = stage3_weight_map_generation(
        target_poses, ref_img_feats, ref_alphas, feature_aggregator_path, device
    )

    ref_alphas = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", B=1)
    weight_maps_heatmap = weight_map_to_heatmap(
        weight_maps, 
        ref_alphas,
        colormap="jet", 
        normalize=True,
        temperature=1.0,
        return_tensor=True
    )

    for i, weight_map_heatmap in enumerate(weight_maps_heatmap):
        weight_map_heatmap = rearrange(
            weight_map_heatmap,
            "B Nr H W C -> (B Nr) C H W",
        )
        save_image(weight_map_heatmap, os.path.join(pred_corr_maps_output_dir, f"{i}.png"))

    print("Stage 4: RGB Generation")
    images = stage4_rgb_generation(
        target_poses, ref_rgbs, weight_maps, base_model_path, rgb_adapter_path, device, dtype
    )

    seg_model = AutoModelForImageSegmentation.from_pretrained(segment_model_name, trust_remote_code=True)
    seg_model.eval()
    seg_model.to(device)

    images_rgba = segment_rgbs(images, seg_model, device)

    for i, image_rgba in enumerate(images_rgba):
        image_rgba.save(os.path.join(pred_rgbs_output_dir, f"{i}.png"))

    print("Finish RGB Inference")

    mv_rgbs = []
    for image_rgba in images_rgba:
        mv_rgbs.append(load_image(image_rgba, 768, 768))
    mv_rgbs = torch.stack(mv_rgbs)
    mv_rgbs = mv_rgbs.permute(0, 3, 1, 2).to(device)

    print("Stage 5: Normal Generation")
    normals = stage5_normal_generation(
        target_poses, mv_rgbs, base_model_path, normal_adapter_path, device, dtype
    )

    normals_rgba = segment_rgbs(normals, seg_model, device)

    for i, normal_rgba in enumerate(normals_rgba):
        normal_rgba.save(os.path.join(pred_normals_output_dir, f"{i}.png"))

    del seg_model
    clear_gpu_memory()

    print("Finish Normal Inference")

    print("Stage 6: Reconstruction")
    gen_obj_path = stage6_reconstruction(
        images_rgba, normals_rgba, smplx_mesh_path, pred_meshes_output_dir, device
    )

    print("Stage 7: Video Rendering")
    stage7_video_rendering(gen_obj_path, output_dir)

    print("Finish Reconstruction")

def main(
    base_model_path,
    normal_adapter_path,
    segment_model_name,
    shape_predictor_path,
    feature_aggregator_path,
    rgb_adapter_path,
    data_dir,
    output_dir,
    dtype,
    device,
):
    os.makedirs(output_dir, exist_ok=True)
    infer_in_the_wild_low_gpu(
        ref_img_dir=data_dir,
        base_model_path=base_model_path,
        rgb_adapter_path=rgb_adapter_path,
        normal_adapter_path=normal_adapter_path,
        feature_aggregator_path=feature_aggregator_path,
        shape_predictor_path=shape_predictor_path,
        segment_model_name=segment_model_name,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    args.add_argument("--shape_predictor_path", type=str, default="pretrained_models/shape_predictor.pt")
    args.add_argument("--rgb_adapter_path", type=str, default="pretrained_models/rgb_adapter.safetensors")
    args.add_argument("--feature_aggregator_path", type=str, default="pretrained_models/feature_aggregator.pt")
    args.add_argument("--normal_adapter_path", type=str, default="pretrained_models/normal_adapter.safetensors")
    args.add_argument("--segment_model_name", type=str, default="ZhengPeng7/BiRefNet")
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--output_dir", type=str, default="outputs")
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()

    dtype = torch.float16
    device = args.device

    base_model_path = args.base_model_path
    normal_adapter_path = args.normal_adapter_path
    segment_model_name = args.segment_model_name
    shape_predictor_path = args.shape_predictor_path

    feature_aggregator_path = args.feature_aggregator_path
    rgb_adapter_path = args.rgb_adapter_path

    data_dir = args.data_dir
    output_dir = args.output_dir

    main(
        base_model_path,
        normal_adapter_path,
        segment_model_name,
        shape_predictor_path,
        feature_aggregator_path,
        rgb_adapter_path,
        data_dir,
        output_dir,
        dtype,
        device,
    )