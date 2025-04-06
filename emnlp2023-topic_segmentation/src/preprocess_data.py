#!/usr/bin/env python3
"""
Vietnamese Wikipedia Segment Formatter (textseg_uet_v1)

This script formats Vietnamese Wikipedia articles into JSONL format for text segmentation tasks.
It tokenizes sentences and marks section boundaries for training segmentation models.

Usage:
    python preprocess_viwiki.py --input /path/to/split_data --output /path/to/processed_output

Dependencies:
    - underthesea for Vietnamese sentence tokenization
    - tqdm for progress bars
"""

import os
import json
import argparse
import configparser
from tqdm import tqdm
from underthesea import sent_tokenize
from analysis.statistics_of_data import data_statistics

# Section flag marker
sec_flag = "========"

def tokenize_method(sec_text):
    """
    Tokenizes section text into sentences and labels the last sentence as the section boundary.
    
    Args:
        sec_text (str): The text content of a section
        
    Returns:
        tuple: (sentences, labels) where labels mark section boundaries (1 for last sentence, 0 otherwise)
    """
    # Get paragraphs
    sec_paragraphs = list(filter(lambda x: x != '', sec_text.split("\n")))
    
    # Tokenize to sentences by underthesea (replaces nltk in the original)
    sec_sents = [sent_tokenize(p) for p in sec_paragraphs]
    
    # Handle empty section case
    if not sec_sents:
        return [], []
    
    # Create labels for sentences
    # Final sentence of topic is 1, other sentences are 0
    sec_sent_labels = [[0] * len(p_sents) for p_sents in sec_sents]
    
    # Flatten sentences and labels
    sec_sents = sum(sec_sents, [])
    sec_sent_labels = sum(sec_sent_labels, [])  # convert to 1-d list
    
    # Mark the last sentence as section boundary
    if sec_sent_labels:
        sec_sent_labels[-1] = 1

    return sec_sents, sec_sent_labels

def process_wiki_folder(folder, out_file):
    """
    Process all text files in a folder.
    
    Args:
        folder (str): Path to folder with text files
        out_file (str): Path to output JSONL file
    """
    # Get all files
    all_files = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.endswith('.txt'):
                all_files.append(os.path.join(root, name))

    examples = []
    for file_ in tqdm(all_files):
        # Get sections sentences and labels
        sentences, labels = [], []
        section_topic_labels = []
        
        with open(file_, "r", encoding='utf-8') as f:
            lines = f.readlines()
            sec_flag_indices = []
            for i, line in enumerate(lines):
                if line.startswith(sec_flag):
                    sec_flag_indices.append(i)
            sec_flag_indices.append(len(lines))

            for i in range(len(sec_flag_indices) - 1):
                start = sec_flag_indices[i] + 1
                end = sec_flag_indices[i + 1]
                if start == end:
                    # Empty section
                    continue
                
                # Extract section title if available
                header_line = lines[sec_flag_indices[i]].strip()
                if ',' in header_line:
                    parts = header_line[len(sec_flag):].split(',', 2)
                    if len(parts) >= 2:
                        section_level = parts[0].strip()
                        section_title = parts[1].strip() if len(parts) == 2 else parts[2].strip()
                        # Remove trailing period if present
                        if section_title.endswith('.'):
                            section_title = section_title[:-1]
                        section_topic_labels.append(section_title)
                
                # Get section content
                sec_text = ''.join(lines[start:end])
                
                # Tokenize section text into sentences and get labels
                sec_sents, sec_labels = tokenize_method(sec_text)
                
                # Skip empty sections
                if not sec_sents:
                    continue
                    
                sentences += sec_sents
                labels += sec_labels
                
        # Create example dictionary
        example = {
            "file": file_,
            "sentences": sentences,
            "labels": labels,
        }
        
        # Add section topics if available
        if section_topic_labels:
            example["section_topic_labels"] = section_topic_labels
            
        # Skip documents with no sentences
        if not sentences:
            continue
            
        examples.append(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("len(examples): ", len(examples))
    with open(out_file, "w", encoding='utf-8') as f:
        f.writelines(examples)
    
    return out_file

def process_textseg_uet_v1(data_folder, out_folder):
    """
    Process textseg_uet_v1 data from train/dev/test folders.
    
    Args:
        data_folder (str): Root folder containing train/dev/test splits
        out_folder (str): Folder to save processed JSONL files
    """
    # Create output folder if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)
    
    # Process each mode (train, dev, test)
    for mode in ["train", "dev", "test"]:
        folder = os.path.join(data_folder, mode)
        if not os.path.exists(folder):
            print(f"Warning: {mode} folder not found at {folder}")
            continue
            
        out_file = os.path.join(out_folder, f"{mode}.jsonl")
        print(f"Processing {mode} data from {folder}")
        print(f"Output will be saved to: {out_file}")
        
        processed_file = process_wiki_folder(folder, out_file)
        
        # Calculate and print statistics
        try:
            data_statistics(out_file)
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            # Fallback statistics calculation
            print_basic_statistics(out_file)

def print_basic_statistics(jsonl_file):
    """
    Print basic statistics for a processed JSONL file (fallback if data_statistics fails).
    
    Args:
        jsonl_file (str): Path to the JSONL file
    """
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        documents = len(lines)
        sentences = 0
        section_boundaries = 0
        
        for line in lines:
            data = json.loads(line)
            sentences += len(data["sentences"])
            section_boundaries += sum(data["labels"])
        
        print(f"\nBasic statistics for {os.path.basename(jsonl_file)}:")
        print(f"  Documents: {documents}")
        print(f"  Total sentences: {sentences}")
        print(f"  Section boundaries: {section_boundaries}")
        print(f"  Average sentences per document: {sentences/documents:.2f}")
        print(f"  Average sections per document: {section_boundaries/documents:.2f}")
        
    except Exception as e:
        print(f"Error calculating basic statistics: {e}")

def get_data_name2folder(config_path):
    """Read data paths from config file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    mapping = config['mapping']
    return mapping

def get_process_dict():
    """Return dictionary of processing functions for different datasets"""
    process_dict = {
        "wiki_section": None,  # Not implemented for Vietnamese
        "wiki50": None,        # Not implemented for Vietnamese
        "wiki727k": None,      # Not implemented for Vietnamese
        "wiki_elements": None, # Not implemented for Vietnamese
        "textseg_uet_v1": process_textseg_uet_v1,
    }
    return process_dict

if __name__ == "__main__":
    # For direct execution with command line arguments
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese Wikipedia articles for text segmentation")
    parser.add_argument("--input", required=True, help="Path to folder containing Wiki article files (with train/dev/test splits)")
    parser.add_argument("--output", required=True, help="Path to save the processed JSONL files")
    parser.add_argument("--data_name", default="textseg_uet_v1", help="Dataset name (default: textseg_uet_v1)")
    
    args = parser.parse_args()
    
    # Get the processing function for the specified dataset
    process_dict = get_process_dict()
    process_fct = process_dict.get(args.data_name)
    
    if process_fct:
        process_fct(args.input, args.output)
    else:
        print(f"Error: Processing function for dataset '{args.data_name}' not implemented")
        
    print("Processing complete!")