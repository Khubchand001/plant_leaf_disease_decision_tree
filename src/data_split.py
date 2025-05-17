import os
import shutil
import random
import argparse
from tqdm import tqdm

def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    class_names = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    for class_name in tqdm(class_names, desc="Splitting dataset"):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_files in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_file in split_files:
                src_path = os.path.join(class_path, img_file)
                dst_path = os.path.join(split_class_dir, img_file)
                if os.path.isfile(src_path):  # only copy files, not directories
                 shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test folders")
    parser.add_argument("--input_dir", type=str, default="data/PlantVillage", help="Input dataset path")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output split dataset path")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
