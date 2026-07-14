# PrismVLA

PrismVLA is being rebuilt around a compact, layer-wise vision-language-action architecture for LIBERO and CALVIN.

The current implementation contract and its explicitly provisional experiment
settings are documented in [Qwen3.5 Query-Bridge Architecture
Baseline](docs/design/qwen35_query_bridge_baseline.md).

The runnable baseline configurations, remote smoke test, checkpoint resume,
and checkpoint-backed policy server are documented in
[Training and serving](docs/training.md).

Benchmark runtime contracts remain documented separately:

- [CALVIN contract](docs/benchmarks/calvin_contract.md)
- [Benchmark runtime](docs/benchmarks/runtime.md)
- [Remote `prsim` environment](docs/environment.md)
