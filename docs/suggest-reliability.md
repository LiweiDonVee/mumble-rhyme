# Suggest-first cuBLAS failure investigation

Date: 2026-10-21. Release: 0.2.2. This investigation used the reported request:

```json
{
  "target": "light",
  "left": "the city lights are",
  "intent": "深夜城市的孤独感",
  "count": 3
}
```

The reported error was `CUBLAS_STATUS_ALLOC_FAILED` during `cublasCreate(handle)`: allocation failure during runtime initialization, rather than an invalid rhyme request.

## Observations and limits

The live Cherry Studio child process ran the default 9B server. After the error it still held approximately 11 GB of private committed process memory; total GPU use was around 10.7 GB including other applications. Windows reported about 2.7 GB of remaining system commit. Only that identified project process was stopped for controlled checks. Other applications and paging settings were left alone.

A fresh production MCP process accepted the same request before the fix. The original native cuBLAS error was therefore **not deterministically reproduced**. The error code alone cannot distinguish device allocation from Windows host-commit pressure at the original failure.

Two concrete gaps were confirmed:

- Previous real MCP checks ranked candidates before suggesting lyrics. They missed generation as the first model operation. A diagnostic run observed about 1.5 GB of additional private commit between loaded weights and first generation.
- Resource errors during generation/scoring propagated out while leaving the model loaded. Cleanup in 0.2.1 covered failed loading only. A unit reproduction using the reported cuBLAS message kept `loaded=true` and temporary allocation-owner frames until this fix.

## Fix

NF4 loading now prepares the current thread's cuBLAS handle and performs an actual small BF16 linear operation before loading large weights. This initializes that path while more memory is available. Unoccupied CUDA loading reservations are returned before inference. It does not guarantee success under unlimited system load.

Scoring and generation now share an allocation-failure boundary. The operation's frames unwind first; chained traceback locals are cleared, weights released, unreachable objects collected, and unused CUDA cache returned. The error retains the original allocation message and says weights were released. Status reports `loaded=false` and `last_resource_error`; later successful inference clears the error. Ordinary validation errors do not discard a healthy model.

There is no automatic retry, fabricated output, or silent switch to 4B. The next caller request can load the same selected model. Some CUDA faults require a process restart; the following checks cover allocation failure, not every driver/kernel fault.

## Reproduce

Run the installed model environment serially:

```console
python scripts/check_suggest_mcp.py --model 9b --output docs/suggest-first-9b.json
python scripts/check_suggest_mcp.py --model 4b --output docs/suggest-first-4b.json
python scripts/check_suggest_mcp.py --model 9b --inject-failure --output docs/suggest-recovery-9b.json
python scripts/check_suggest_mcp.py --model 4b --inject-failure --output docs/suggest-recovery-4b.json
```

The first two start the normal production `serve` entrypoint, send the reported suggestion before any scoring operation, then repeat it in the same process. Both calls must return candidates with actual model scores. As in the API contract, count is a proposal budget; duplicate proposals can leave fewer options, and requested/received counts are returned.

The latter two use a **test-only child bootstrap**. It allocates a temporary 16 MiB CUDA tensor in the actual model's first generation and raises the reported allocation error. This is not a claim of native driver-error reproduction. The client checks the error, `loaded=false`, allocated CUDA memory below 128 MiB, and two subsequent successful requests in the same server. No fault-injection switch exists in the production server.

Records: [9B cold-first](suggest-first-9b.json), [4B cold-first](suggest-first-4b.json), [9B recovery](suggest-recovery-9b.json), [4B recovery](suggest-recovery-4b.json). These are functional and resource-lifetime checks, not an editorial quality evaluation.

## Updating a client

A running MCP process keeps its imported code. After updating, stop/restart the server in Cherry Studio's MCP settings and call `model_status`. `package_version` must be `0.2.2` or later. Changing size still requires an explicit startup option such as `--model 4b`.

The old failed child was stopped before opening separate test servers to avoid two 9B copies. Test servers exit when their clients close. The application must start a new MCP connection after updating.
