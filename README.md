# Benchmark

## SymbolicAI: A framework for logic-based approaches combining generative models and solvers

We introduce SymbolicAI, a versatile and modular framework employing a logic-based approach to concept learning and flow management in generative processes. SymbolicAI enables the seamless integration of generative models with a diverse range of solvers by treating large language models (LLMs) as semantic parsers that execute tasks based on both natural and formal language instructions, thus bridging the gap between symbolic reasoning and generative AI. We leverage probabilistic programming principles to tackle complex tasks, and utilize differentiable and classical programming paradigms with their respective strengths. The framework introduces a set of polymorphic, compositional, and self-referential operations for data stream manipulation, aligning LLM outputs with user objectives. As a result, we can transition between the capabilities of various foundation models endowed with zero- and few-shot learning capabilities and specialized, fine-tuned models or solvers proficient in addressing specific problems. In turn, the framework facilitates the creation and evaluation of explainable computational graphs. We conclude by introducing a quality measure and its empirical score for evaluating these computational graphs, and propose a benchmark that compares various state-of-the-art LLMs across a set of complex workflows. We refer to the empirical score as the "Vector Embedding for Relational Trajectory Evaluation through Cross-similarity", or VERTEX score for short. The SymbolicAI framework codebase is available [here](https://github.com/ExtensityAI/symbolicai).

## Installation

### Requirements

Install dependencies.

```bash
pip install "symbolicai[all]"
pip install -r requirements.txt
```

Install LlamaCpp backend.

```bash
sympkg i ExtensityAI/llamacpp
```

Then follow the instructions in the [ExtensityAI/llamacpp](https://github.com/ExtensityAI/llamacpp) repository to install and run the LlamaCpp backend with various HuggingFace models.

Install embeddings backend.

```bash
sympkg i ExtensityAI/embeddings
```

## Usage

Run the full benchmark.

```bash
python test.py --context_associations --program_synthesis --multimodal_bindings --logic_components --computation_graphs
```

This will run all the evaluations in the benchmark.

## Cite us

```bibtex
@article{
    Dinu:24,
    title={SymbolicAI: A framework for logic-based approaches combining generative models and solvers},
    author={Marius–Constantin Dinu and Claudiu Leoveanu–Condrei and Markus Holzleitner and Werner Zellinger and Sepp Hochreiter},
    year={2024},
    eprint={2402.00854},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    url={https://arxiv.org/abs/2402.00854}
}
```
