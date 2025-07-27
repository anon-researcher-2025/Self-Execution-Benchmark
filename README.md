# Self-Awareness Project

## Overview

This project investigates the self-awareness capabilities of Large Language Models (LLMs) through a series of comprehensive experiments. The research explores how well AI models can recognize their own outputs, predict their responses, and understand their capabilities and limitations.

## Project Structure

The project consists of four main experiments, each targeting different aspects of LLM self-awareness:

### Experiment 1: Word Association Prediction
**Directory:** `experiment 1/`

Tests whether models can predict which words they would associate with a given target word. The experiment:
- Asks models to predict if specific words would appear in their association lists
- Then generates actual word associations
- Compares predictions with actual outputs to measure self-awareness

**Key Files:**
- `main.py` - Single-threaded implementation
- `main_thread_support.py` - Multi-threaded implementation  
- `3-5_taboo_words.json` - Dataset of target words and associated options
- `create_dataset.py` - Dataset generation utilities

### Experiment 2 - Restriction Recognition Test
**Directory:** `experiment 2/`

Evaluates whether models can recognize their own restrictions.

**Key Files:**
- `GPTAPI.py` - Single model implementation
- `GPTAPI_parallel.py` - Parallel processing for multiple models
- `xstest_prompts.csv` - Test prompts dataset
- `dataprocessing.ipynb` - Results analysis notebook

### Experiment 3 – Difficulty Assessment Test
**Directory:** `experiment 3/`

Analyzes models' ability to predict the relative difficulty of questions and rank them by solving time.

**Key Files:**
- `TimeTest.py` - Core experiment implementation
- `TimeTest_parallel.py` - Parallel processing version
- `correlation_analysis.py` - Statistical analysis of results
- `model_accuracy_analysis.py` - Performance evaluation
- Uses MMLU and CommonSenseQA datasets

### Experiment 4: Self-Generated Question Performance
**Directory:** `experiment 4/`

Tests whether models can create questions they expect to answer incorrectly, then evaluates their actual performance on these self-generated questions.

**Key Files:**
- `ex4.py` - Full experiment implementation
- `ex4_test.py` - Test version with fewer models
- `experiment_4_analysis.ipynb` - Results analysis

### Additional Components

#### LLM Response Recognition Studies
**Directory:** `llm_recognize_his_answers/`

Extended studies on response recognition with various prompt formats and methodologies.

#### Data Processing and Analysis
- `data_procrsor.ipynb` - Main data processing notebook
- `plotting.py` - Visualization utilities
- `utilities.py` - Shared utility functions

#### Benchmarking
**Directory:** `benchmarks/`

Contains benchmark datasets and comparison metrics for model evaluation.

## Supported Models

The project evaluates multiple state-of-the-art language models:

### OpenAI Models
- GPT-4o
- GPT-4o-mini

### OpenRouter Models
- Meta LLaMA (3.1-8B, 3.2-3B)
- DeepSeek (Chat, R1)
- Qwen 2.5 (7B, 32B)
- Mistral (7B, Small 3.1-24B, Mixtral 8x22B)
- Google Gemini (Pro, 2.0 Flash)
- Anthropic Claude (3.5 Haiku, 3.7 Sonnet)

## Key Features

- **Multi-model Evaluation**: Comprehensive testing across diverse model architectures
- **Parallel Processing**: Efficient batch processing for large-scale experiments
- **Statistical Analysis**: Correlation analysis and performance metrics
- **Visualization**: Comprehensive plotting and data visualization
- **Reproducible Research**: Structured experiments with detailed logging

## Setup and Installation

1. **Clone the repository:**
```bash
git clone <path to the github repository>
cd Self-Awareness-Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
or you can run `run_enviroment.sh` or `run_enviroment.bat` to install all dependencies and run enviroment. 

3. **Set up API keys:**
   - Create `openai_key.txt` with your OpenAI API key
   - Update `api_keys.txt` with OpenRouter credentials

4. **Configure environment:**
   - Run `run_environment.sh` (Linux/Mac) or `run_environment.bat` (Windows)

## Usage

### Running Individual Experiments

**Experiment 1 - Word Association:**
```bash
cd "experiment 1"
python main.py
```

**Experiment 2 - Response Recognition:**
```bash
cd "experiment 2"
python GPTAPI_parallel.py
```

**Experiment 3 - Difficulty Ranking:**
```bash
cd "experiment 3"
python TimeTest_parallel.py
```

**Experiment 4 - Self-Generated Questions:**
```bash
cd "experiment 4"
python ex4.py
```

### Multi-Model Processing
Use `multy pro.py` for running experiments across multiple models simultaneously:
```bash
python "multy pro.py"
```

### Data Analysis
Process and visualize results using the Jupyter notebooks:
```bash
jupyter notebook data_procrsor.ipynb
```

## Results and Analysis

Results are saved in CSV format in respective experiment directories. The project includes:
- Statistical correlation analysis
- Performance benchmarking
- Visualization of self-awareness metrics
- Cross-model comparison studies

## Research Applications

This project contributes to understanding:
- LLM metacognitive capabilities
- Model reliability and self-assessment
- AI safety and trustworthiness
- Cognitive architectures in artificial systems


 
