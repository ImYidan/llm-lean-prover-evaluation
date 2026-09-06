# LLM Lean Prover Evaluation

This repository contains the technical evaluation pipeline that accompanies
my master's project on LLM-based Lean theorem proving. The code runs
Goedel-Prover-V2-32B, DeepSeek-Prover-V2-7B CoT,
Kimina-Prover-Distill-8B, and Pythagoras-Prover-4B, preserves raw responses,
assembles Lean
submissions, checks them with Lean, and reports empirical Pass@k results.

Start with [technical/README.md](technical/README.md) for setup, command-line
interfaces, output schemas, and benchmark-specific pipelines.
For DeepSeek-Prover-V2-7B CoT, read
[docs/deepseek-7b-cot.md](docs/deepseek-7b-cot.md).
The [historical DeepSeek archive](archives/deepseek-prover-v2-7b/README.md)
preserves the original evaluation implementation, input fingerprints, dependency
pins and its differences from the shared pipeline.
The autoregressive releases are documented in
[docs/kimina-prover-distill-8b.md](docs/kimina-prover-distill-8b.md) and
[docs/pythagoras-prover-4b.md](docs/pythagoras-prover-4b.md).

You must supply benchmark data, model weights, Lean workspaces, and writable
output paths. This repository does not distribute datasets, weights, logs,
checkpoints, or experiment result artifacts.

## License

The code uses the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE)
for upstream attribution.
