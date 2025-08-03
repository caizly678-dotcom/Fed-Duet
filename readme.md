# FedDuet: A Dual-channel Expert-orchestrated Framework for Federated Continual Learning

[![Paper-ARXIV-Version](https://img.shields.io/badge/Paper-ARXIV-b31b1b.svg)](https://arxiv.org/abs/YOUR_PAPER_ID) <!-- TODO: Add your paper link -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- TODO: Choose and confirm license -->

This is the official PyTorch implementation for the paper **"FedDuet: A Dual-channel Expert-orchestrated Framework for Federated Continual Learning in Vision-Language Models"**.

---

## Abstract

Pretrained vision-language models (VLMs) like CLIP have significantly enhanced federated learning by bringing strong representation capabilities to edge devices. However, in practical edge scenarios, dynamic task shifts and heterogeneous data distributions undermine the continual learning capability of vision-language models. Despite recent efforts, existing parameter-efficient fine-tuning (PEFT) approaches effectively reduce communication costs for large-scale models but fail to maintain satisfactory performance as tasks evolve over time. Meanwhile, traditional federated continual learning (FCL) methods struggle to achieve effective multimodal alignment necessary for robust vision-language integration. To address the above challenges, we propose **Fed-Duet**, a novel **Du**al-channel **E**xper**t**-orchestrated framework for efficient federated continual learning in vision-language models. Fed-Duet employs a dual-channel architecture that orchestrates two complementary experts: 1) *guiding prompts* for semantic alignment and 2) *modular adapters* for task specialization, with their outputs dynamically integrated to maintain coherence and adaptability across evolving tasks. Extensive experiments validate the superior performance and robustness of Fed-Duet on challenging continual learning tasks in federated vision-language settings. Our work bridges the gap between VLM adaptation and lifelong learning in federated scenarios, paving the way for scalable multimodal intelligence.

## Framework

The core of FedDuet is its dual-channel architecture, which processes information through two complementary pathways before dynamically integrating their outputs.

```mermaid
graph TD
    A[Input <br/> (Images/Text)] --> B{Vision-Language Model <br/> (e.g., CLIP Backbone)};
    B --> C[FedDuet Experts];
    subgraph C [ ]
        direction LR
        C1[Channel 1: <br/> Guiding Prompts] --> O1[Semantic Alignment Features];
        C2[Channel 2: <br/> Modular Adapters] --> O2[Task-Specific Features];
    end
    O1 --> D{Dynamic Integration};
    O2 --> D;
    D --> E[Final Output <br/> (Prediction)];

    style C fill:#f9f,stroke:#333,stroke-width:2px
```

## Getting Started

### 1. Installation

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/FedDuet.git
cd FedDuet
```

We recommend creating a virtual environment to manage dependencies:
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

Install the required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**Note on PyTorch**: The `requirements.txt` file specifies PyTorch versions. If you need a different version that matches your specific CUDA setup, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command.

### 2. Dataset Preparation

This project uses several datasets for evaluation, including CIFAR-100, Tiny ImageNet, and DomainNet.

Please download the datasets and place them in a directory of your choice. You will need to specify the path to your dataset directory in the corresponding configuration files located in `cil/configs/`.

<!-- TODO: Add more specific instructions on dataset structure if necessary -->
```
/path/to/your/datasets/
├── cifar-100-python/
├── tiny-imagenet-200/
└── domain_net/
```

Update the `dataset_root` variable in the `.yaml` configuration files to point to `/path/to/your/datasets/`.

## Running Experiments

The main script for running experiments is `cil/run.sh`. This script uses Hydra for configuration management, allowing for flexible experiment setups.

To run a default experiment, you can execute the script directly:
```bash
bash cil/run.sh
```

### Customizing Runs

You can customize experiments by modifying the configuration files located in `cil/configs/708/FedDuet/`. For example, you can change:
- The dataset (`dataset`)
- The number of clients (`num_clients`)
- The number of communication rounds (`com`)
- The learning rate (`lr`)

The `run.sh` script is pre-configured to run multiple experiments. You can uncomment or modify the desired configuration paths and names within the script to reproduce specific results from our paper.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{your_name_2024_fedduet,
  title={FedDuet: A Dual-channel Expert-orchestrated Framework for Federated Continual Learning in Vision-Language Models},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

<!-- TODO: Update with your official BibTeX entry -->

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

<!-- TODO: Confirm the license choice and add a LICENSE file. --> 