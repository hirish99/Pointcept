import argparse
import torch
import numpy as np
import open3d as o3d
from pointcept.models import build_model
from pointcept.utils.config import get_cfg


def load_ply(ply_path):
    """Load a .ply file into numpy arrays"""
    pcd = o3d.io.read_point_cloud(ply_path)
    coords = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        feats = np.asarray(pcd.colors, dtype=np.float32)
    else:
        feats = np.ones_like(coords, dtype=np.float32)  # dummy features
    return coords, feats


def normalize_points(coords):
    """Center and scale to unit sphere"""
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / scale
    return coords


def voxelize(coords, feats, voxel_size=0.02):
    """Downsample point cloud with voxel grid filter"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(feats)

    pcd = pcd.voxel_down_sample(voxel_size)
    coords = np.asarray(pcd.points, dtype=np.float32)
    feats = np.asarray(pcd.colors, dtype=np.float32)
    return coords, feats


def save_ply(coords, labels, save_path, palette):
    """Save point cloud with predicted labels as colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    colors = np.array([palette[l] for l in labels]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)


def get_palette(num_classes):
    """Generate a simple color palette"""
    np.random.seed(0)
    return (np.random.rand(num_classes, 3) * 255).astype(np.int32)


def main(args):
    # --- load config and model ---
    cfg = get_cfg(args.config)
    model = build_model(cfg.model).cuda().eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- load scan ---
    coords, feats = load_ply(args.input)

    # normalize + voxelize
    coords = normalize_points(coords)
    coords, feats = voxelize(coords, feats, voxel_size=args.voxel_size)

    coord = torch.from_numpy(coords).unsqueeze(0).cuda()  # [1, N, 3]
    feat = torch.from_numpy(feats).unsqueeze(0).cuda()    # [1, N, 3]

    # --- inference ---
    with torch.no_grad():
        outputs = model({"coord": coord, "feat": feat})
        preds = outputs["pred"].argmax(1).squeeze().cpu().numpy()

    # --- save results ---
    palette = get_palette(cfg.model.num_classes)
    save_ply(coords, preds, args.output, palette)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pointcept Single Scan Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input .ply file")
    parser.add_argument("--output", type=str, default="predicted.ply", help="Output .ply with predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to model config .yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint .pth")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for downsampling")
    args = parser.parse_args()
    main(args)
