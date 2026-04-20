import os
import json
from pathlib import Path
from tqdm import tqdm


# TODO: Change DATA_ROOT to the path of your downloaded data folder
DATA_ROOT = "Downloaded Data Folder"

DATA_DIR = [
    f"{DATA_ROOT}/2K2K",
    f"{DATA_ROOT}/4D-DRESS",
    f"{DATA_ROOT}/CustomHumans",
    f"{DATA_ROOT}/Human4DiT",
    f"{DATA_ROOT}/THuman2.1",
    f"{DATA_ROOT}/X_Humans",
]

VIEWS_6 = {"000", "045", "090", "180", "270", "315"}


def _view_sort_key(path: str):
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def _collect_view_paths(view_dir: str):
    view_paths = []
    for view_name in os.listdir(view_dir):
        view_path = os.path.join(view_dir, view_name)
        if not os.path.isfile(view_path):
            continue
        if Path(view_name).stem not in VIEWS_6:
            continue
        view_paths.append(view_path)
    return sorted(view_paths, key=_view_sort_key)


def collect_data(
    data_dir,
    output_dir="data",
):
    all_data = []
    id_names = sorted(os.listdir(data_dir))
    for id_name in tqdm(id_names, desc=f"Processing {data_dir}"):
        id_dir = os.path.join(data_dir, id_name)
        id_data = {
            "id": id_name,
            "source_paths": {
                "front_top": [],
                "front_bottom": [],
                "back_top": [],
                "back_bottom": [],
                "left_top": [],
                "left_bottom": [],
                "right_top": [],
                "right_bottom": [],    
            },
            "cases_target_data": []
        }


        case_names = sorted(os.listdir(id_dir))
        for case_name in case_names:
            case_dir = os.path.join(id_dir, case_name)
            if not os.path.isdir(case_dir):
                continue

            source_image_dir = os.path.join(case_dir, "source")
            id_data["source_paths"]["front_top"].append(os.path.join(source_image_dir, "front_top.png"))
            id_data["source_paths"]["front_bottom"].append(os.path.join(source_image_dir, "front_bottom.png"))
            id_data["source_paths"]["back_top"].append(os.path.join(source_image_dir, "back_top.png"))
            id_data["source_paths"]["back_bottom"].append(os.path.join(source_image_dir, "back_bottom.png"))
            id_data["source_paths"]["left_top"].append(os.path.join(source_image_dir, "left_top.png"))
            id_data["source_paths"]["left_bottom"].append(os.path.join(source_image_dir, "left_bottom.png"))
            id_data["source_paths"]["right_top"].append(os.path.join(source_image_dir, "right_top.png"))
            id_data["source_paths"]["right_bottom"].append(os.path.join(source_image_dir, "right_bottom.png"))
            
            target_mesh_normal_dir = os.path.join(case_dir, "target_mesh_normal")
            target_mesh_rgb_dir = os.path.join(case_dir, "target_mesh_rgb")
            target_smplx_normal_dir = os.path.join(case_dir, "target_smplx_normal")
            target_smplx_params_path = os.path.join(case_dir, "smplx", "smplx_params.pkl")

            case_target_data = {
                "case_name": case_name,
                "target_mesh_normal_paths": _collect_view_paths(target_mesh_normal_dir),
                "target_mesh_rgb_paths": _collect_view_paths(target_mesh_rgb_dir),
                "target_smplx_normal_paths": _collect_view_paths(target_smplx_normal_dir),
                "target_smplx_params_path": target_smplx_params_path,
            }
            
            id_data["cases_target_data"].append(case_target_data)
        all_data.append(id_data)

    dataset_name = os.path.basename(data_dir)
    with open(os.path.join(output_dir, f"{dataset_name}.json"), "w") as f:
        json.dump(all_data, f)
        
if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    for data_dir in DATA_DIR:
        collect_data(data_dir, output_dir)
