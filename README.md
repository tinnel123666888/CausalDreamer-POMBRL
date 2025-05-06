
# CausalDreamer-POMBRL

Official implementation of the paper:

**Causal Dreamer for Partially Observable Model-Based Reinforcement Learning**  
(*Under Review / Submitted to [Your Conference or Journal Name]*)  

## 🚀 Overview

This repository contains the code and experimental setup for our proposed **Causal Dreamer** framework, which improves model-based reinforcement learning (MBRL) in partially observable environments by integrating counterfactual inference for history compression and long-term reasoning.

Key contributions:

- A causality-based model that filters historical observations via **history counterfactual inference**.
- A **coarse-to-fine intervention strategy** that reduces computational overhead in causal mining.
- A variant of the Dreamer algorithm enhanced with causal memory selection, achieving superior performance in long-horizon, memory-dependent tasks.

## 📋 Paper

coming soon.  


## 🧠 Method Summary

We extend the Dreamer framework with a **causal gate** mechanism that filters out irrelevant history using a self-supervised auxiliary task based on counterfactual interventions. The resulting model improves the efficiency and accuracy of dynamics modeling and policy learning.

## 🛠️ Setup

### Requirements

* Python 3.8+
* PyTorch >= 1.10
* NumPy
* \[Any other major dependencies]

```bash
pip install -r requirements.txt
```

### Folder Structure

```
.
├── agents/                  # Core RL agent and Dreamer modifications
├── envs/                    # Partially observable task environments
├── scripts/                 # Training and evaluation scripts
├── utils/                   # Helper modules
├── configs/                 # YAML configs for training runs
└── README.md
```

## 🧪 Running Experiments

To train Causal Dreamer on BabyAI or Maze:

```bash
python scripts/train.py --config configs/babyai.yaml
```

To evaluate a trained model:

```bash
python scripts/eval.py --checkpoint path/to/model.ckpt
```

## 📊 Results

## 📊 Results

Below are the learning curves of **Causal Dreamer** and baseline methods across all sub-tasks:

<img src="curves_all_tasks.png" alt="Learning Curves for All Tasks" width="100%"/>


## 📎 License

MIT License. See `LICENSE` for details.

## 🙏 Acknowledgements

This codebase builds on [DreamerV2](https://github.com/danijar/dreamerv2) and [BabyAI](https://github.com/mila-iqia/babyai).

```


