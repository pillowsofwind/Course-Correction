# Code for Reproduction for "Course-Correction: Safety Alignment Using Synthetic Preferences"

## Description:

This repo consists of code for reproducing the main results of "Course-Correction: Safety Alignment Using Synthetic Preferences".

## eval Folder - Corresponding to C2-EVAL in the Paper

Use `eval_data.py` to generate evaluation data, and `gpteval.py` to evaluate using GPT-4o.

## syn Folder - Corresponding to C2-SYN in the Paper

The `train.jsonl` file is the training dataset from [PKU-Alignment](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), please download it and put it in the `syn` folder.
Use `create_data.py` to generate datasets and `process_data.py` to process data to meet the DPO format.

