# EatVul-Resources: SHAP-Enhanced Defense Against Adversarial Attac## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training and Evaluation
```bash
# Train and evaluate on a specific dataset (e.g., asterisk)
python ./run_model.sh asterisk

#Make sure run_model.sh is executable
chmod +x run_model.sh

```

### 3. Adversarial Code Generation
```bash
# Generate base snippets
python file/code/generate_base_snippets.py

# Generate adversarial samples
python file/code/adversarial_code_generation.py

# Run FGA selection
python file/code/fga_selection.py
```

## 📁 Dataset Structure

The `data/` folder contains:
- `*_train.json`: Training datasets
- `*_test.json`: Clean test datasets
- `*_test_ADV.json`: Adversarial test datasets (size=1)

Available datasets:
- Asterisk
- CWE119
- CWE399
- OpenSSL

## 🔧 Code Components

### Core Modules
1. **Model Implementation** (`model/`)
   - `saab_model_run.py`: BiLSTM+SVM implementation
   - `analyze_shap.py`: SHAP value analysis
   - `shap_defense.py`: Defense mechanism
   - `test_adversarial_generator.py`: Attack testing

2. **Adversarial Generation** (`file/code/`)
   - `adversarial_code_generation.py`: ChatGPT-based generation
   - `fga_selection.py`: FGA seed sample selection
   - `generate_base_snippets.py`: Base snippet creation
   - `improved_adversarial_generation.py`: Enhanced generations in Vulnerability Detection

This paper was accepted by USENIX Security '24. The source code and datasets provided are for research use only and not for commercial use.

## 🎯 Project Overview

This repository implements a robust defense mechanism against adversarial attacks in vulnerability detection systems, using SHAP-based analysis and a hybrid BiLSTM+SVM architecture. Building upon the original EaTVul framework, we've enhanced the system with defensive capabilities.
## 🌟 Key Features and Performance

### Model Architecture
- BiLSTM encoder with attention mechanism
- SVM classifier with probability calibration
- SHAP-based defense mechanism
- Support for multiple vulnerability datasets

### Performance Highlights

#### Clean Test Performance
| Dataset   | Accuracy | Attack Success Rate |
|-----------|----------|-------------------|
| Asterisk  | 94.55%   | 10.00%           |
| CWE119    | 87.88%   | 7.07%            |
| CWE399    | 85.10%   | 20.41%           |

#### SHAP Defense Results
| Dataset   | Detection Rate | Original Accuracy | Final Accuracy |
|-----------|---------------|-------------------|----------------|
| Asterisk  | 88.00%        | 44.00%           | 64.00%         |
| CWE119    | 100.00%       | 86.00%           | 100.00%        |
| CWE399    | 80.50%        | 55.50%           | 80.50%         |

## 🛠 Requirements and Setup

### Dependencies
```bash
# Core ML stack
numpy>=1.24,<2.0
scikit-learn>=1.4,<1.6
torch>=2.6,<2.9

# Explainability
shap>=0.45,<0.48

# Utilities
pyyaml>=6.0.1
regex>=2023.6.3
```

### System Requirements
- Python 3.13 or later
- 8GB RAM (minimum)
- GPU optional but recommended
- Unix-like system (Linux/macOS preferred)

## 📁 Project Structure

```
project_root/
├── file/
│   ├── code/              # Adversarial code generation
│   └── data/             # Dataset files
├── model/                # Core implementations
│   ├── saab_model_run.py    # Main model runner
│   ├── shap_defense.py      # Defense mechanism
│   └── analyze_shap.py      # SHAP analysis
└── runs/                 # Output directory

To fine-tune the target model (in this case, CodeBERT works as an example.)via the following command:
```
python ori_model_run.py\
  --output_dir=./saved_newbap_models/model\
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --train_data_file=./example_train.json \
  --eval_data_file=./example_test.json \
  --test_data_file=./example_test.json \
  --epoch 10 \
  --block_size 400 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456 2>&1 | tee train.log
```  
To train the surrogate model with knowledge distillation, please run the follwoing commands:
```
python surrogate_train.py
```
To evaluate the performance of the surrogate model, run the following commands:
```
python surrogate_test.py
```
The "data" folder conatins the data samples used in this paper to validate the findings, where "xxxxx_train.json" is the train dataset and "xxxxx_test.json" works as the test dataset. And the “xxxxx_test_ADV.json” is the testcase set with adversarial code snippets (with code snippet size of 1).  

The "code" folder contains two functional modules:  
&emsp;&emsp;1. Generate adversarial code snippets with chatGPT. -- adversarial_code_generation.py  
&emsp;&emsp;2. Use FGA to select the seed adversarial samples. -- fga_selection.py  


Remark: The data file suffixed with ‘train’ is used for training the model, and the file suffixed with ‘adv’ is used as the adversarial file.
Python version: Python3.7


