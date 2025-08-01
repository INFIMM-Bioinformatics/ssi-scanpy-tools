# Scanpy Tools

**Scanpy Tools** is a Python library for single-cell RNA sequencing analysis that provides various tools for preprocessing, visualization, and analysis of single-cell data. It leverages the [Scanpy](https://scanpy.readthedocs.io/) library and offers a comprehensive and user-friendly API.

## Features

- Preprocessing of single-cell RNA sequencing data
- Visualization tools for single-cell data
- Comprehensive analysis tools
- User-friendly API

## Installation

To install Scanpy Tools, you can use pip:

```bash
#pip install scanpy-tools
```

### AI module

#### Conda

```bash
# Use conda to install vLLM
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

#### Serve the model

```bash
# check whether /work/HF_cache exists
#mkdir -p /work/HF_cache
#export HF_HOME=/work/HF_cache
conda activate vllm

# ✅ 1.5B - DeepSeek-R1-Distill-Qwen-1.5B
# ✅ 7B - DeepSeek-R1-Distill-Qwen-7B
# ✅ 8B - DeepSeek-R1-Distill-Llama-8B
# 14B - DeepSeek-R1-Distill-Qwen-14B (biggest model one can run on a H100 GPU)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --max-model-len 32768 --enforce-eager

```

### Setting up a Development Environment

To set up a virtual environment for development, follow these steps:

#### Create a Virtual Environment
```bash
python3 -m venv .venv
```

#### Activate the Virtual Environment
```bash
source .venv/bin/activate
```

#### Upgrade pip
```bash
pip install --upgrade pip
```

#### Install the Package in Development Mode
```bash
pip install -e .
```

#### Install Additional Dependencies
```bash
pip install scanpy jupyter matplotlib plotly pandas
```

After completing these steps, you will have a fully functional development environment for working on Scanpy Tools.

## Contributing
We welcome contributions! Please see our contributing guidelines for more information.

## License
This project is licensed under the MIT License. See the LICENSE file for details.