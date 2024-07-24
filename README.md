# Course-Correction: Safety Alignment Using Synthetic Preferences

## Description:

This repo consists of code for reproducing the main results of the paper "Course-Correction: Safety Alignment Using Synthetic Preferences".

```
@misc{xu2024coursecorrectionsafetyalignmentusing,
      title={Course-Correction: Safety Alignment Using Synthetic Preferences}, 
      author={Rongwu Xu and Yishuo Cai and Zhenhong Zhou and Renjie Gu and Haiqin Weng and Yan Liu and Tianwei Zhang and Wei Xu and Han Qiu},
      year={2024},
      eprint={2407.16637},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16637}, 
}
```

## Evaluating course-correction performance

> eval Folder - Corresponding to C$^2$-EVAL in the Paper

1. Please first use the script in `eval_data.py` to generate evaluation data.
2. Then use `gpteval.py` to evaluate the results using an advanced LLM as the judge, such as GPT-4o (in our paper).

## Learning to course-correct using synthetic preferences

> syn Folder - Corresponding to C$^2$-SYN in the Paper

1. The `train.jsonl` file is the training dataset from [PKU-Alignment](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), please download it and put it in the `syn` folder.
2. Use `create_data.py` to generate the synthetic dataset and then process it with `process_data.py` to get the standard pairwise preference format.
3. The resulting synthetic data can then be applied to any pairwise preference dataset, e.g., DPO, PPO, etc.

