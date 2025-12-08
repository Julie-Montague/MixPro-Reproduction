#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

def _is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    return str(target).startswith(str(directory) + os.sep)

def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    path = path.resolve()
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt in tar member: {member.name}")
    tar.extractall(path)

def extract_tar(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        safe_extract(tar, dst_dir)

def load_devkit_mapping(devkit_tar_gz: Path, tmp_dir: Path) -> Tuple[List[str], List[int]]:
    """
    Returns:
      wnids: list of 1000 synset strings, index 0..999 corresponds to label 1..1000
      val_labels: list of 50000 ints in [1..1000] (ground truth order matches val image filename order)
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extract_tar(devkit_tar_gz, tmp_dir)

    # Typical path after extraction:
    # tmp_dir/ILSVRC2012_devkit_t12/data/meta.mat
    # tmp_dir/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
    devkit_root = next(tmp_dir.glob("**/ILSVRC2012_devkit_t12"), None)
    if devkit_root is None:
        raise FileNotFoundError("Could not find ILSVRC2012_devkit_t12 folder inside devkit tarball.")

    meta_mat = devkit_root / "data" / "meta.mat"
    gt_txt = devkit_root / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not meta_mat.exists() or not gt_txt.exists():
        raise FileNotFoundError(f"Missing meta.mat or validation ground truth file in devkit: {devkit_root}")

    # Read labels
    val_labels = [int(x.strip()) for x in gt_txt.read_text().splitlines() if x.strip()]
    if len(val_labels) != 50000:
        raise RuntimeError(f"Expected 50000 validation labels, got {len(val_labels)}")

    # Read wnid mapping from meta.mat via scipy
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy is required to parse meta.mat. Install with: pip install scipy") from e

    mat = loadmat(meta_mat)
    synsets = mat.get("synsets", None)
    if synsets is None:
        raise RuntimeError("meta.mat did not contain 'synsets' variable as expected.")

    # Build mapping ILSVRC2012_ID -> WNID for 1..1000
    # This parsing is intentionally defensive because MATLAB structs load oddly.
    id_to_wnid: Dict[int, str] = {}

    synsets_flat = synsets.ravel()
    for s in synsets_flat:
        try:
            # Typical struct fields: ['ILSVRC2012_ID', 'WNID', 'words', ...]
            ilsvrc_id = int(s["ILSVRC2012_ID"][0][0])
            wnid = str(s["WNID"][0])
            if 1 <= ilsvrc_id <= 1000:
                id_to_wnid[ilsvrc_id] = wnid
        except Exception:
            continue

    if len(id_to_wnid) != 1000:
        raise RuntimeError(f"Expected 1000 wnids, got {len(id_to_wnid)}. meta.mat parsing failed.")

    wnids = [id_to_wnid[i] for i in range(1, 1001)]
    return wnids, val_labels

def unpack_train(train_tar: Path, out_train: Path, tmp_dir: Path) -> None:
    """
    train tar contains 1000 class tar files (wnid.tar). We extract each into train/wnid/.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Extracting top-level train tar to tmp: {tmp_dir}")
    extract_tar(train_tar, tmp_dir)

    class_tars = sorted(tmp_dir.glob("*.tar"))
    if len(class_tars) < 900:
        print(f"[warn] Expected ~1000 class tars, found {len(class_tars)}. Check train tar integrity.")

    out_train.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for idx, ctar in enumerate(class_tars, start=1):
        wnid = ctar.stem
        dst = out_train / wnid
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ctar, "r:*") as tar:
            safe_extract(tar, dst)
        if idx % 50 == 0:
            dt = time.time() - t0
            print(f"[train] {idx}/{len(class_tars)} classes unpacked... ({dt:.1f}s)")

def unpack_val_and_sort(val_tar: Path, out_val: Path, wnids: List[str], val_labels: List[int], tmp_dir: Path) -> None:
    """
    val tar contains 50k images in a single folder. We extract, then move into val/wnid/ folders.
    """
    tmp_images = tmp_dir / "val_images"
    tmp_images.mkdir(parents=True, exist_ok=True)

    print(f"[val] Extracting val tar to tmp: {tmp_images}")
    extract_tar(val_tar, tmp_images)

    # Ensure deterministic order by filename
    images = sorted(tmp_images.glob("*.JPEG"))
    if len(images) != 50000:
        raise RuntimeError(f"Expected 50000 val images, got {len(images)}")

    out_val.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, img_path in enumerate(images):
        label = val_labels[i]  # 1..1000
        wnid = wnids[label - 1]
        dst_dir = out_val / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        img_path.replace(dst_dir / img_path.name)
        if (i + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f"[val] {i+1}/50000 images sorted... ({dt:.1f}s)")

def count_jpegs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.JPEG"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw/imagenet",
                    help="Directory containing ILSVRC tar files.")
    ap.add_argument("--out_dir", type=str, default="data/imagenet1k",
                    help="Output directory (will create train/ and val/).")
    ap.add_argument("--tmp_dir", type=str, default="data/tmp/imagenet_extract",
                    help="Temporary directory for extraction.")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_val", action="store_true")
    ap.add_argument("--keep_tmp", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    tmp = Path(args.tmp_dir)

    train_tar = raw / "ILSVRC2012_img_train.tar"
    val_tar = raw / "ILSVRC2012_img_val.tar"
    devkit_tar = raw / "ILSVRC2012_devkit_t12.tar.gz"

    missing = [p.name for p in [train_tar, val_tar, devkit_tar] if not p.exists()]
    if missing and not (args.skip_train or args.skip_val):
        raise FileNotFoundError(f"Missing required files in {raw}: {missing}")

    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    meta_path = out / "meta.json"
    if not meta_path.exists():
        wnids, val_labels = load_devkit_mapping(devkit_tar, tmp / "devkit")
        meta = {"wnids": wnids, "num_classes": 1000}
        meta_path.write_text(json.dumps(meta, indent=2))
        (out / "val_labels.txt").write_text("\n".join(map(str, val_labels)))
        print(f"[meta] Wrote {meta_path} and val_labels.txt")
    else:
        meta = json.loads(meta_path.read_text())
        wnids = meta["wnids"]
        val_labels = [int(x.strip()) for x in (out / "val_labels.txt").read_text().splitlines() if x.strip()]

    if not args.skip_train:
        print("[train] Preparing train set...")
        unpack_train(train_tar, out / "train", tmp / "train")
        print(f"[train] Done. JPEG count: {count_jpegs(out / 'train')}")
    else:
        print("[train] Skipped.")

    if not args.skip_val:
        print("[val] Preparing val set...")
        unpack_val_and_sort(val_tar, out / "val", wnids, val_labels, tmp)
        print(f"[val] Done. JPEG count: {count_jpegs(out / 'val')}")
    else:
        print("[val] Skipped.")

    if not args.keep_tmp:
        # cautious cleanup: only delete tmp_dir we created
        import shutil
        print(f"[tmp] Removing tmp dir: {tmp}")
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"Train dir: {out/'train'}")
    print(f"Val dir:   {out/'val'}")

if __name__ == "__main__":
    main()
