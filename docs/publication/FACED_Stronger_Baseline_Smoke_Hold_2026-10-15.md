# FACED continuation: manufactured CUDA smoke hold

15 October 2026. The continuation frozen at `cb936ffbe` **stopped in its manufactured smoke before any new source replay or activation**. There is still no completed source EEGNet learning-capacity or held-source utility result. Keep both original and continuation attempts intact.

The explicit `cuda:0` adapter passed device selection, model construction, forward inference and loss calculation. The first manufactured `loss.backward()` then raised `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`, before `optimizer.step()`, epoch completion or checkpoint saving. The retained smoke receipt has `passed:false`, zero checkpoint records, zero source-signal/cache-target reads and `protected_access:false`. No source-only activation or actual execution file was produced for this continuation.

Read-only runtime inspection reports Torch2.5.1, compiled CUDA12.4, cuDNN90100 and an RTX3050 6GB laptop GPU. NVIDIA-SMI reports driver592.82 and765/6144MiB occupied after the failed process ended. This is not a measurement of memory at failure and does not diagnose the cause. No process was killed, driver/package modified, CUDA backend changed or retry launched.

One bounded diagnostic hypothesis concerns batch geometry: the smoke used four fabricated training trials (28 flattened windows), whereas every declared real training minibatch has eight trials (56 windows), because32tiny/448main training trials are divisible by8. Prior successful hardware preflight also used original-trial batch8. A separately frozen representative-eight-trial smoke could test that implementation geometry without changing a real-data schedule. The hypothesis is not evidence that batch geometry caused this error, and selecting a scientific model/target from a successful hardware configuration would be inappropriate.

The source research scope, paper, candidate set, all ten closed confirmation gates and reserved outcomes remain unchanged. Any further manufactured diagnostic or continuation needs a fresh bounded declaration and one-attempt accounting; the failed smoke must remain reported separately from actual-data fits.
