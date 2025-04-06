#!/usr/bin/env python3
"""
Vietnamese Wikipedia Dataset Splitter

This script splits a collection of Vietnamese Wikipedia articles into
train, dev, and test sets. It ensures a stratified split based on article
size (number of sections) to maintain similar distributions across sets.

Usage:
    python split_viwiki_dataset.py --input /path/to/viwiki_data --output /path/to/split_output [--train 0.8] [--dev 0.1] [--test 0.1]

"""

import os
import shutil
import random
import argparse
from collections import defaultdict
import json
from tqdm import tqdm

def count_sections(file_path):
    """Count the number of sections in a Vietnamese Wikipedia article file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Count section markers (========,)
            return content.count("========,")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def get_size_category(section_count):
    """Categorize articles by size based on section count"""
    if section_count <= 3:
        return "small"
    elif section_count <= 6:
        return "medium"
    else:
        return "large"

def split_viwiki_dataset(input_folder, output_folder, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split Vietnamese Wikipedia dataset into train, dev, and test sets.
    
    Args:
        input_folder (str): Path to folder containing Wiki article files
        output_folder (str): Path to save the split datasets
        train_ratio (float): Proportion for training set (default: 0.8)
        dev_ratio (float): Proportion for development set (default: 0.1)
        test_ratio (float): Proportion for test set (default: 0.1)
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Statistics about the split
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_folder, "train")
    dev_dir = os.path.join(output_folder, "dev")
    test_dir = os.path.join(output_folder, "test")
    
    for directory in [train_dir, dev_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Find all text files
    all_files = []
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.txt'):
                all_files.append(os.path.join(root, filename))
    
    print(f"Found {len(all_files)} text files in {input_folder}")
    
    # Group files by size category for stratified sampling
    files_by_category = defaultdict(list)
    category_counts = defaultdict(int)
    
    print("Analyzing file sizes...")
    for file_path in tqdm(all_files):
        section_count = count_sections(file_path)
        category = get_size_category(section_count)
        files_by_category[category].append((file_path, section_count))
        category_counts[category] += 1
    
    # Print category statistics
    print("\nFile category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} files ({count/len(all_files)*100:.1f}%)")
    
    # Split each category according to the ratios
    train_files = []
    dev_files = []
    test_files = []
    
    for category, files in files_by_category.items():
        # Shuffle files within category
        random.shuffle(files)
        
        # Calculate split indices
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_dev = int(n_files * dev_ratio)
        
        # Split files
        train_files.extend(files[:n_train])
        dev_files.extend(files[n_train:n_train+n_dev])
        test_files.extend(files[n_train+n_dev:])
    
    # Copy files to their respective directories
    split_stats = {
        "total": len(all_files),
        "train": len(train_files),
        "dev": len(dev_files),
        "test": len(test_files),
        "by_category": {}
    }
    
    # Function to copy files for a specific split
    def copy_files_to_split(file_list, target_dir, split_name):
        split_categories = defaultdict(int)
        
        print(f"\nCopying {len(file_list)} files to {split_name} set...")
        for file_info, section_count in tqdm(file_list):
            filename = os.path.basename(file_info)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(file_info, target_path)
            
            category = get_size_category(section_count)
            split_categories[category] += 1
        
        return split_categories
    
    # Copy files for each split
    split_stats["by_category"]["train"] = dict(copy_files_to_split(train_files, train_dir, "train"))
    split_stats["by_category"]["dev"] = dict(copy_files_to_split(dev_files, dev_dir, "dev"))
    split_stats["by_category"]["test"] = dict(copy_files_to_split(test_files, test_dir, "test"))
    
    # Save statistics to a JSON file
    stats_file = os.path.join(output_folder, "split_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(split_stats, f, indent=2)
    
    # Print final statistics
    print("\nDataset split complete!")
    print(f"Total files: {len(all_files)}")
    print(f"Training set: {len(train_files)} files ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"Development set: {len(dev_files)} files ({len(dev_files)/len(all_files)*100:.1f}%)")
    print(f"Test set: {len(test_files)} files ({len(test_files)/len(all_files)*100:.1f}%)")
    print(f"\nDetailed statistics saved to {stats_file}")
    
    return split_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Vietnamese Wikipedia dataset into train/dev/test sets")
    parser.add_argument("--input", required=True, help="Path to folder containing Wiki article files")
    parser.add_argument("--output", required=True, help="Path to save the split datasets")
    parser.add_argument("--train", type=float, default=0.8, help="Proportion for training set (default: 0.8)")
    parser.add_argument("--dev", type=float, default=0.1, help="Proportion for development set (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Proportion for test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train + args.dev + args.test - 1.0) > 0.001:
        parser.error("Split ratios must sum to 1.0")
    
    # Run the split
    split_viwiki_dataset(
        args.input, 
        args.output, 
        train_ratio=args.train, 
        dev_ratio=args.dev, 
        test_ratio=args.test,
        seed=args.seed
    )