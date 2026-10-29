# WMP Dreamer source

Imported on 2026-10-26 from the user's local `wmp_go1-main/dreamer` snapshot.
The source files retain their MIT and Bytedance attribution notices. The
repository-level Apache-2.0 license is included as `LICENSE.upstream.txt`.

Original source SHA256:

| File | SHA256 |
| --- | --- |
| models.py | ed3b876e40604c8eec324550af19d400ba4655f276415e4bec06c5134ec57dc0 |
| networks.py | d1118d2ce7392b779152470fb3c45e2801b92df8aaa5d7f05a7232886e81f9c0 |
| tools.py | 5e7daafd202f018cecaf73554f783f6d76787388ae96148e339e45161dc488ce |
| configs.yaml | 8fb8cd42c6e0f9058f965b1799f0423742adc52487638e26416523ad9a6d52ae |

`models.py`, `networks.py` and `configs.yaml` are imported without algorithm
changes. `tools.py` has only these local runtime safeguards:

- `torch.cuda.amp.GradScaler` becomes `torch.amp.GradScaler("cuda", ...)` for
  current PyTorch. The default FP32 optimizer has scaling disabled as upstream.
- Non-finite scalar losses and gradient norms fail before Adam changes weights.

Do not reformat these vendored files. Lint/type checks apply to the adapter, not
upstream's unused environment/logging helpers. Compare upstream before changing
the network or losses. `world_model.py` outside this directory maps replay uint8
images to the reference's normalized floats and maps local dimension names to
the YAML defaults. `compute_loss` is a validation-only copy of the upstream
forward objective; training calls upstream `WorldModel._train` directly.
