# Benchmark

## SymbolicAI: A framework for logic-based approaches combining generative models and solvers

We introduce SymbolicAI, a versatile and modular framework that employs a logic-based approach to concept learning and flow management in generative processes. This framework facilitates the seamless integration of generative models with a diverse range of solvers. We integrate large language models (LLMs) and probabilistic programming principles to tackle complex tasks and utilize differentiable and classical programming paradigms with their respective strengths. Central to our framework are neuro-symbolic engines, which include LLMs for task executions of natural and formal language instructions. Our approach establishes a set of  operations that manipulate a data stream and guide the LLMs to align with the user's objectives. Inherently, these operations exhibit polymorphic, compositional and self-referential properties, which facilitate the implementation of various data types and behaviors. As a result, we can seamlessly transition between capabilities of various foundation models endowed with zero and few-shot learning capabilities and specialized, fine-tuned models or solvers proficient in addressing specific problems. In turn, our framework enables the creation of explainable computational graphs through compositional expressions and functions.
Our code and benchmark are available for further exploration here, and at our [ExtensityAI/symbolicai](https://github.com/ExtensityAI/symbolicai) GitHub page.

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
python test.py --context_associations --program_synthesis --multimodal_bindings --components --computation_graphs
```

This will run all the evaluations in the benchmark.

## Cite us

```bibtex
@article{
    Dinu:24,
    title={SymbolicAI: A framework for logic-based approaches combining generative models and solvers},
    author={Marius–Constantin Dinu and Claudiu Leoveanu–Condrei and Eric Mitchell and Christopher D Manning and Stefano Ermon and Sepp Hochreiter},
    year={2024},
    url={https://arxiv.org/abs/TODO}
}
```