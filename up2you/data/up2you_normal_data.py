import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.config import parse_structured
from ..utils.typing import *


VIEWS_6 = {"000", "045", "090", "180", "270", "315"}
VIEWS_4 = {"000", "090", "180", "270"}


@dataclass
class MultiviewPuzzleDataModuleConfig:
    data_dir: Any
    validation_data_dir: Optional[Any] = None

    ref_background_color: Union[str, float] = "gray"
    target_background_color: Union[str, float] = "gray"
    train_num_views: int = 6
    val_num_views: int = 6
    default_prompt: Optional[str] = None

    used_data_type: Optional[Tuple[str, str]] = None
    ref_mask_aug_ratio: float = 0.3

    repeat: int = 1

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    target_height: int = 768
    target_width: int = 768

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16


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


def _parse_one_json_scene(json_path: str, num_views: int = 6):
    scene_list = []
    with open(json_path, "r") as f:
        json_data = json.load(f)

    for id_data in json_data:
        for case_data in id_data["cases_target_data"]:
            scene = {
                "target_image_paths": _filter_target_view_paths(
                    case_data["target_mesh_rgb_paths"], num_views
                ),
                "target_normal_camera_paths": _filter_target_view_paths(
                    case_data["target_mesh_normal_paths"], num_views
                ),
                "target_pose_camera_paths": _filter_target_view_paths(
                    case_data["target_smplx_normal_paths"], num_views
                ),
            }
            scene_list.append(scene)

    return scene_list


def _parse_scene_list(data_paths: Any, num_views: int = 6):
    scene_list = []
    for json_path in _ensure_path_list(data_paths):
        scene_list.extend(_parse_one_json_scene(json_path, num_views))
    return scene_list


class MVPuzzleDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiviewPuzzleDataModuleConfig = cfg
        self.split = split

        if split == "train":
            data_paths = self.cfg.data_dir
            num_views = self.cfg.train_num_views
        else:
            data_paths = self.cfg.validation_data_dir or self.cfg.data_dir
            num_views = self.cfg.val_num_views

        self.all_scenes = _parse_scene_list(data_paths, num_views)
        random.shuffle(self.all_scenes)

        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.train_indices[0] : self.cfg.train_indices[1]]
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.val_indices[0] : self.cfg.val_indices[1]]
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_scenes = self.all_scenes[self.cfg.test_indices[0] : self.cfg.test_indices[1]]

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
        mask_aug: bool = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        image = image.resize((width, height))
        image = torch.from_numpy(np.array(image)).float() / 255.0

        alpha = image[:, :, 3]

        if mask_aug:
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

        image = image[:, :, :3] * image[:, :, 3:4] + background_color * (1 - image[:, :, 3:4])
        return image

    def load_default_prompt(self, default_prompt: str = "3D Human, Realistic, High Quality, HDR"):
        if self.cfg.default_prompt is None:
            return default_prompt
        return self.cfg.default_prompt

    def __getitem__(self, index):
        rgb_background_color = torch.as_tensor(self.get_bg_color(self.cfg.ref_background_color))
        target_background_color = torch.as_tensor(self.get_bg_color(self.cfg.target_background_color))
        scene_dir = self.all_scenes[index]

        out_dir = {
            "num_views": self.cfg.train_num_views if self.split == "train" else self.cfg.val_num_views,
            "prompts": [self.default_prompt],
        }

        if self.used_data_type is not None and "target_rgbs" in self.used_data_type:
            target_rgbs = [
                self.load_image(
                    target_image_path,
                    height=self.cfg.target_height,
                    width=self.cfg.target_width,
                    background_color=rgb_background_color,
                    mask_aug=random.random() < self.ref_mask_aug_ratio,
                )
                for target_image_path in scene_dir["target_image_paths"]
            ]
            out_dir["target_rgbs"] = torch.stack(target_rgbs, dim=0).permute(0, 3, 1, 2)

        if self.used_data_type is not None and "target_normals_camera" in self.used_data_type:
            target_normals_camera = [
                self.load_image(
                    target_normal_path,
                    height=self.cfg.target_height,
                    width=self.cfg.target_width,
                    background_color=target_background_color,
                )
                for target_normal_path in scene_dir["target_normal_camera_paths"]
            ]
            out_dir["target_normals_camera"] = torch.stack(target_normals_camera, dim=0).permute(0, 3, 1, 2)

        if self.used_data_type is not None and "target_poses_camera" in self.used_data_type:
            target_poses_camera = [
                self.load_image(
                    target_pose_path,
                    height=self.cfg.target_height,
                    width=self.cfg.target_width,
                    background_color=target_background_color,
                )
                for target_pose_path in scene_dir["target_pose_camera_paths"]
            ]
            out_dir["target_poses_camera"] = torch.stack(target_poses_camera, dim=0).permute(0, 3, 1, 2)

        return out_dir

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])

        num_views = self.cfg.train_num_views if self.split == "train" else self.cfg.val_num_views

        for k in batch.keys():
            if k in ["target_rgbs", "target_normals_camera", "target_poses_camera"]:
                batch[k] = pack(batch[k])
        for k in ["prompts"]:
            batch[k] = [item for pair in zip(*batch[k]) for item in pair]

        batch.update(
            {
                "num_views": num_views,
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
            self.val_dataset = MVPuzzleDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MVPuzzleDataset(self.cfg, "test")

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
