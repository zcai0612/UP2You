import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ..utils.typing import *
from ..utils.config import parse_structured

@dataclass
class MultiviewPuzzleDataModuleConfig:
    data_dir: Any
    validation_data_dir: Any

    ref_background_color: Union[str, float] = "gray"
    target_background_color: Union[str, float] = "gray"
    train_num_views: int = 6
    train_num_refs: int = 8
    val_num_views: int = 6
    val_num_refs: int = 8
    default_prompt: Optional[str] = None

    used_data_type: Optional[Tuple[str, str]] = None
    ref_mask_aug_ratio: float = 0.0
    
    repeat: int = 1  # for debugging purpose

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    target_height: int = 768
    target_width: int = 768

    ref_height: int = 512
    ref_width: int = 512

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16

SOURCE_ORDER = ["front_top", "front_bottom", "back_top", "back_bottom", "left_top", "left_bottom"]
VIEWS_6 = {"000", "045", "090", "180", "270", "315"}
VIEWS_4 = {"000", "090", "180", "270"}


def _ensure_path_list(data_paths: Any) -> List[str]:
    if isinstance(data_paths, str):
        return [data_paths]
    if isinstance(data_paths, Path):
        return [str(data_paths)]
    try:
        return list(data_paths)
    except TypeError:
        pass
    return [data_paths]


def _view_sort_key(path: str):
    stem = _view_name(path)
    try:
        return int(stem)
    except ValueError:
        return stem


def _view_name(path: str) -> str:
    return Path(path).stem


def _filter_target_view_paths(view_paths: List[str], num_views: int) -> List[str]:
    sorted_paths = sorted(view_paths, key=_view_sort_key)
    if num_views == 6:
        sorted_paths = [path for path in sorted_paths if _view_name(path) in VIEWS_6]
    if num_views == 4:
        sorted_paths = [path for path in sorted_paths if _view_name(path) in VIEWS_4]
    return sorted_paths


def _sample_source_refs_from_groups(source_paths: Dict[str, List[str]], num_refs: int) -> List[str]:
    grouped_paths = {}
    for key, paths in source_paths.items():
        if paths:
            grouped_paths[key] = paths.copy()
            random.shuffle(grouped_paths[key])

    selected = []
    indices = {key: 0 for key in SOURCE_ORDER}
    order_idx = 0
    max_steps = max(1, num_refs * max(1, len(SOURCE_ORDER)))

    while len(selected) < num_refs and order_idx < max_steps:
        key = SOURCE_ORDER[order_idx % len(SOURCE_ORDER)]
        if key in grouped_paths and indices[key] < len(grouped_paths[key]):
            selected.append(grouped_paths[key][indices[key]])
            indices[key] += 1
        order_idx += 1

    if len(selected) < num_refs:
        fallback_paths = []
        for key, paths in grouped_paths.items():
            start_idx = indices.get(key, 0)
            fallback_paths.extend(paths[start_idx:])
        random.shuffle(fallback_paths)
        selected.extend(fallback_paths[: num_refs - len(selected)])

    return selected[:num_refs]


def _parse_scene_list_train(train_data_dir: List[str], num_refs: int, num_views: int=6):
    train_scene_list = []
    for json_path in _ensure_path_list(train_data_dir):
        train_scene_list.extend(_parse_one_json_scene_train(json_path, num_refs, num_views))
    return train_scene_list


def _parse_scene_list_val(val_data_dir: List[str], num_refs: int, num_views: int=6):
    val_scene_list = []
    for json_path in _ensure_path_list(val_data_dir):
        val_scene_list.extend(_parse_one_json_scene_val(json_path, num_refs, num_views))
    return val_scene_list


def _parse_one_json_scene_val(json_path: str, num_refs: int, num_views: int=6):
    scene_list = []
    with open(json_path, "r") as f:
        json_data = json.load(f)

    for id_data in json_data:
        ref_image_paths = _sample_source_refs_from_groups(id_data["source_paths"], num_refs)
        for case_data in id_data["cases_target_data"]:
            scene = {}
            scene["target_pose_camera_paths"] = _filter_target_view_paths(
                case_data["target_smplx_normal_paths"], num_views
            )
            scene["ref_image_paths"] = ref_image_paths.copy()
            scene_list.append(scene)

    return scene_list


def _parse_one_json_scene_train(json_path: str, num_refs: int, num_views: int=6):
    scene_list = []
    with open(json_path, "r") as f:
        json_data = json.load(f)

    for id_data in json_data:
        ref_image_paths = _sample_source_refs_from_groups(id_data["source_paths"], num_refs)
        for case_data in id_data["cases_target_data"]:
            scene = {}
            scene["target_image_paths"] = _filter_target_view_paths(
                case_data["target_mesh_rgb_paths"], num_views
            )
            scene["target_pose_camera_paths"] = _filter_target_view_paths(
                case_data["target_smplx_normal_paths"], num_views
            )
            scene["ref_image_paths"] = ref_image_paths.copy()
            scene_list.append(scene)

    return scene_list

class ValDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "val") -> None:
        super().__init__()
        assert split in ["val", "test"]
        self.cfg: MultiviewPuzzleDataModuleConfig = cfg
        self.val_scenes = _parse_scene_list_val(self.cfg.validation_data_dir, self.cfg.val_num_refs, self.cfg.val_num_views)

        self.all_scenes = self.val_scenes
        random.shuffle(self.all_scenes)

        if split == "val" and self.cfg.val_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.val_indices[0]:self.cfg.val_indices[1]]
        elif split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.test_indices[0]:self.cfg.test_indices[1]]

        self.default_prompt = self.load_default_prompt()

        self.used_data_type = self.cfg.used_data_type

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    def load_image(
        self,
        image: Union[str, Image.Image],
        height: int,
        width: int,
        background_color: torch.Tensor,
        rescale: bool = False,
        mask_aug: bool = False,
        return_alpha: bool = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.resize((width, height))
        image = torch.from_numpy(np.array(image)).float() / 255.0

        alpha = image[:, :, 3]

        if mask_aug:
            # alpha = image[:, :, 3]  # Extract alpha channel
            h, w = alpha.shape
            y_indices, x_indices = torch.where(alpha > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                idx = torch.randint(len(y_indices), (1,)).item()
                y_center = y_indices[idx].item()
                x_center = x_indices[idx].item()
                mask_h = random.randint(h // 8, h // 4)
                mask_w = random.randint(w // 8, w // 4)

                y1 = max(0, y_center - mask_h // 2)
                y2 = min(h, y_center + mask_h // 2)
                x1 = max(0, x_center - mask_w // 2)
                x2 = min(w, x_center + mask_w // 2)

                alpha[y1:y2, x1:x2] = 0.0
                image[:, :, 3] = alpha

        image = image[:, :, :3] * image[:, :, 3:4] + background_color * (
            1 - image[:, :, 3:4]
        )
        if rescale:
            image = image * 2.0 - 1.0
        if return_alpha:
            return image, alpha
        else:
            return image
    
    def load_default_prompt(self, default_prompt: str="3D Human, Realistic, High Quality, HDR"):
        if self.cfg.default_prompt is None:
            return default_prompt
        else:
            return self.cfg.default_prompt
        
    def __getitem__(self, index):
        ref_background_color = torch.as_tensor(self.get_bg_color(self.cfg.ref_background_color))
        target_background_color = torch.as_tensor(self.get_bg_color(self.cfg.target_background_color))
        scene_dir = self.all_scenes[index]

        ref_image_paths = scene_dir["ref_image_paths"]
        target_pose_camera_paths = scene_dir["target_pose_camera_paths"]

        out_dir = {}
        out_dir["num_views"] = self.cfg.val_num_views
        out_dir["num_refs"] = self.cfg.val_num_refs
        out_dir["prompts"] = [self.default_prompt]

        ref_images = []
        ref_alphas = []

        if self.used_data_type is not None and "ref_rgbs" in self.used_data_type:
            for ref_image_path in ref_image_paths:
                ref_image, ref_alpha = self.load_image(
                    ref_image_path, 
                    height=self.cfg.ref_height, 
                    width=self.cfg.ref_width, 
                    background_color=ref_background_color,
                    return_alpha=True,
                )
                ref_images.append(ref_image)
                ref_alphas.append(ref_alpha)
            ref_images = torch.stack(ref_images, dim=0).permute(0, 3, 1, 2)
            ref_alphas = torch.stack(ref_alphas, dim=0)

            out_dir["ref_rgbs"] = ref_images
            out_dir["ref_alphas"] = ref_alphas

        if self.used_data_type is not None and "target_poses_camera" in self.used_data_type:
            target_poses_camera = [
                self.load_image(
                    target_pose_path, 
                    height=self.cfg.target_height, 
                    width=self.cfg.target_width, 
                    background_color=target_background_color,
                )
                for target_pose_path in target_pose_camera_paths
            ]
            target_poses_camera = torch.stack(target_poses_camera, dim=0).permute(0, 3, 1, 2)
            out_dir["target_poses_camera"] = target_poses_camera

        return out_dir
    
    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])

        num_views = self.cfg.val_num_views
        num_refs = self.cfg.val_num_refs

        for k in batch.keys():
            if k in ["ref_rgbs", "ref_alphas", "target_poses_camera"]:
                batch[k] = pack(batch[k])
        for k in ["prompts"]:
            batch[k] = [item for pair in zip(*batch[k]) for item in pair]

        batch.update(
            {
                "num_views": num_views,
                "num_refs": num_refs,
                # For SDXL
                "original_size": (self.cfg.target_height, self.cfg.target_width),
                "target_size": (self.cfg.target_height, self.cfg.target_width),
                "crops_coords_top_left": (0, 0),
            }
        )
        return batch

class MVPuzzleDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiviewPuzzleDataModuleConfig = cfg
        self.all_scenes = _parse_scene_list_train(self.cfg.data_dir, self.cfg.train_num_refs, self.cfg.train_num_views)
        random.shuffle(self.all_scenes)
        self.split = split
        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.train_indices[0]:self.cfg.train_indices[1]]
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.val_indices[0]:self.cfg.val_indices[1]]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.test_indices[0]:self.cfg.test_indices[1]]

        self.default_prompt = self.load_default_prompt()
        self.used_data_type = self.cfg.used_data_type

        self.ref_mask_aug_ratio = self.cfg.ref_mask_aug_ratio

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color
    
    def load_image(
        self,
        image: Union[str, Image.Image],
        height: int,
        width: int,
        background_color: torch.Tensor,
        rescale: bool = False,
        mask_aug: bool = False,
        return_alpha: bool = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.resize((width, height))
        image = torch.from_numpy(np.array(image)).float() / 255.0   

        alpha = image[:, :, 3]

        if mask_aug:
            # alpha = image[:, :, 3]  # Extract alpha channel
            h, w = alpha.shape
            y_indices, x_indices = torch.where(alpha > 0.5)
            if len(y_indices) > 0 and len(x_indices) > 0:
                idx = torch.randint(len(y_indices), (1,)).item()
                y_center = y_indices[idx].item()
                x_center = x_indices[idx].item()
                mask_h = random.randint(h // 8, h // 4)
                mask_w = random.randint(w // 8, w // 4)

                y1 = max(0, y_center - mask_h // 2)
                y2 = min(h, y_center + mask_h // 2)
                x1 = max(0, x_center - mask_w // 2)
                x2 = min(w, x_center + mask_w // 2)

                alpha[y1:y2, x1:x2] = 0.0
                image[:, :, 3] = alpha

        image = image[:, :, :3] * image[:, :, 3:4] + background_color * (
            1 - image[:, :, 3:4]
        )
        if rescale:
            image = image * 2.0 - 1.0
        if return_alpha:
            return image, alpha
        else:
            return image
    
    def load_default_prompt(self, default_prompt: str="3D Human, Realistic, High Quality, HDR"):
        if self.cfg.default_prompt is None:
            return default_prompt
        else:
            return self.cfg.default_prompt
        
    def __getitem__(self, index):
        ref_background_color = torch.as_tensor(self.get_bg_color(self.cfg.ref_background_color))
        target_background_color = torch.as_tensor(self.get_bg_color(self.cfg.target_background_color))
        scene_dir = self.all_scenes[index]

        ref_image_paths = scene_dir["ref_image_paths"]
        target_image_paths = scene_dir["target_image_paths"]
        target_pose_camera_paths = scene_dir["target_pose_camera_paths"]

        out_dir = {}
        out_dir["num_views"] = self.cfg.train_num_views
        out_dir["num_refs"] = self.cfg.train_num_refs
        out_dir["prompts"] = [self.default_prompt]

        ref_images = []
        ref_alphas = []
        if self.used_data_type is not None and "ref_rgbs" in self.used_data_type:
            for ref_image_path in ref_image_paths:
                ref_image, ref_alpha = self.load_image(
                    ref_image_path, 
                    height=self.cfg.ref_height, 
                    width=self.cfg.ref_width, 
                    background_color=ref_background_color,
                    return_alpha=True,
                    mask_aug=random.random() < self.ref_mask_aug_ratio,
                )
                ref_images.append(ref_image)
                ref_alphas.append(ref_alpha)
            ref_images = torch.stack(ref_images, dim=0).permute(0, 3, 1, 2)
            ref_alphas = torch.stack(ref_alphas, dim=0)
            out_dir["ref_rgbs"] = ref_images
            out_dir["ref_alphas"] = ref_alphas

        target_images = []
        target_alphas = []
        if self.used_data_type is not None and "target_rgbs" in self.used_data_type:
            for target_image_path in target_image_paths:
                target_image, target_alpha = self.load_image(
                    target_image_path, 
                    height=self.cfg.target_height, 
                    width=self.cfg.target_width, 
                    background_color=target_background_color,
                    return_alpha=True,
                )
                target_images.append(target_image)
                target_alphas.append(target_alpha)
            target_images = torch.stack(target_images, dim=0).permute(0, 3, 1, 2)
            target_alphas = torch.stack(target_alphas, dim=0)
            out_dir["target_rgbs"] = target_images
            out_dir["target_alphas"] = target_alphas

        if self.used_data_type is not None and "target_poses_camera" in self.used_data_type:
            target_poses_camera = [
                self.load_image(
                    target_pose_path, 
                    height=self.cfg.target_height, 
                    width=self.cfg.target_width, 
                    background_color=target_background_color,
                )
                for target_pose_path in target_pose_camera_paths
            ]
            target_poses_camera = torch.stack(target_poses_camera, dim=0).permute(0, 3, 1, 2)
            out_dir["target_poses_camera"] = target_poses_camera

        return out_dir
    
    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])

        num_views = self.cfg.train_num_views
        num_refs = self.cfg.train_num_refs

        for k in batch.keys():
            if k in ["ref_rgbs", "ref_alphas", "target_rgbs", "target_alphas", "target_poses_camera"]:
                batch[k] = pack(batch[k])
        for k in ["prompts"]:
            batch[k] = [item for pair in zip(*batch[k]) for item in pair]

        batch.update(
            {
                "num_views": num_views,
                "num_refs": num_refs,
                # For SDXL
                "original_size": (self.cfg.target_height, self.cfg.target_width),
                "target_size": (self.cfg.target_height, self.cfg.target_width),
                "crops_coords_top_left": (0, 0),
            }
        )
        return batch
    
    
class MultiviewPuzzleDataModule(pl.LightningDataModule):
    cfg: MultiviewPuzzleDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewPuzzleDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MVPuzzleDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ValDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ValDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
