# BkdAtk-LWS
Source code for the ACL 2021 paper "Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution" [[pdf](https://arxiv.org/pdf/2106.06361)]

## Getting started

- If you don't have PyTorch installed, install it here: https://pytorch.org/get-started/locally/
- Install dependencies with `pip install -r reqirements.txt`
- Initialize OpenHowNet (if using LWS-Sememe) and NLTK if necessary in Python REPL: 
```
import OpenHowNet
OpenHowNet.download()
import nltk
nltk.download('all')
```

## Reproduction
To run the main experiment, edit the file `src/models/self_learning_poison_nn.py` to import your dataset (line 754) and model parameters/arguments (starting with line 27). Then, run `python -m src.models.self_learning_poison_nn.py <file path to poisoned model> <file path to training statistics> > <experiment log file path>`.

- Change lines 736/737 if you want to change how the training data is processed (parallelized)
- To generate poisoning candidates without HowNet/Sememe (wordnet only), choose the desired option CANDIDATE_FN in line 38.

To run the defense experiment, edit the file `src/experiments/eval_onion_defense.py` and run `python -m src.experiments.eval_onion_defense.py <location of poisoned model> > <experiment log file path>`.

To run the baseline experiments:

- Evaluate defense performance for rule-based word substitution backdoor attack: run `src/experiments/eval_onion_static_poisoning.py`

## Citation

Please kindly cite our paper:

```
@article{qi2021turn,
  title={Turn the combination lock: Learnable textual backdoor attacks via word substitution},
  author={Qi, Fanchao and Yao, Yuan and Xu, Sophia and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2106.06361},
  year={2021}
}
```