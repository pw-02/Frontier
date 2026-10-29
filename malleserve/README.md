# MalleServe in Frontier

Use this **top-level MalleServe interface** for research experiments. Existing scripts under `examples/malleserve` remain supported to avoid breaking older commands. The next cleanup can consolidate the legacy calibration and trace conversion scripts after parity testing.

## Interactive, in-process debugging

Open `malleserve/debug_run.py` in VS Code/PyCharm, set `CONFIG` to a YAML file, optionally edit `OVERRIDES`, put a breakpoint inside `frontier/simulator.py` (or `frontier/transient_lifecycle.py`), and select **Debug Python File**. Or run:

```bash
python -m malleserve.debug_run
```

Unlike the older shell scenario runner, this calls Frontier's actual `frontier.main.main()` in the **same Python interpreter**, so breakpoints in the native scheduler and DES event loop are available. Requires the Frontier Python environment and your trace/manifest. The default `naive.yaml` uses the 330-second Mooncake rate-10 trace; edit CONFIG to `malleserve/configs/fixed4.yaml` for fixed four GPUs. Runtime profiles come from the checked-in calibrated A100/Qwen3-8B paths.

## Presets

- `configs/fixed4.yaml` fixed four replicas.
- `configs/fixed8.yaml` fixed eight replicas.
- `configs/naive.yaml` four persistent plus rotating transient four-GPU nodes.

These currently inherit the existing presets so there is one source of truth for scenario parameters. The top-level Python runner is intended for step-through debugging; the historical `examples/malleserve/transient/run_native_scenario.py` remains available for CLI runs.

Workload traces are produced by `examples/malleserve/convert_mooncake.py`, which consumes prepared Mooncake requests and writes a CSV of request arrival timestamps, prompt/decode token lengths and cache-block hashes along with an adjacent manifest. Transient node availability is **not** the same trace; `debug_run.py` generates its availability JSONL separately from the YAML resource lifetime. Changing the resource policy but retaining the same workload CSV gives comparable experiments.

**Status**: Python runner added but still requires native execution and parity checks in the Frontier environment. It does not implement native ideal migration.
