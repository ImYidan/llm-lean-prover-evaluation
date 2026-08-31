# LLM Lean Prover Evaluation

This repository contains the technical evaluation pipeline that accompanies
my master's project on LLM-based Lean theorem proving. The code runs
Goedel-Prover-V2-32B and DeepSeek-Prover-V2-7B CoT, preserves raw responses,
assembles Lean
submissions, checks them with Lean, and reports empirical Pass@k results.

Start with [technical/README.md](technical/README.md) for setup, command-line
interfaces, output schemas, and benchmark-specific pipelines.
For DeepSeek-Prover-V2-7B CoT, read
[docs/deepseek-7b-cot.md](docs/deepseek-7b-cot.md).

You must supply benchmark data, model weights, Lean workspaces, and writable
output paths. This repository does not distribute datasets, weights, logs,
checkpoints, or experiment result artifacts.

## License

The code uses the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for upstream attribution.
