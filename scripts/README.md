# Experiment Entry Points

The public entry points are intentionally small and manifest-driven. Dataset-specific conversion scripts should write the internal query and candidate formats described in `docs/data_protocol.md`; training and evaluation should then consume those manifests without reading ground truth at inference time.

Current baseline command:

```bash
python -m src.scalar_reranker \
  --train /path/to/train_manifest.jsonl \
  --output /path/to/scalar_reranker.pt
```

The implementation is being added in the same order as the experiment plan: protocol first, then scalar baseline, then pairwise and relational ranking.

