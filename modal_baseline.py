"""B0 on Modal (handoff §6.1): the upstream SAG2 loop and the SGLang judge server in one L40S container.

  export MODAL_TOKEN_ID=$modal MODAL_TOKEN_SECRET=$modal_secret
  modal run --detach modal_baseline.py --name s1-smoke --dataset musique --limit 10 --expansions 2
  modal volume get sag2poc-a-results s1-smoke results/modal/

The container holds a copy of the pinned upstream tree with patches/0001-jevlike-backend.patch applied; the
submodule itself is never modified. Weights come from the jevlike volume (read only here, never committed).
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

FP8_REPO, FP8_REV = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "ed35d7abe5d9"
GPU = os.environ.get("SAG2POC_GPU", "L40S")
GPU_USD_PER_HOUR = {"L40S": 1.95, "H100": 3.95}

app = modal.App("sag2poc-a-baseline")
weights = modal.Volume.from_name("gb10-decide-hf")
results = modal.Volume.from_name("sag2poc-a-results", create_if_missing=True)

UPSTREAM_IGNORE = [".venv", ".cache", "output", "**/__pycache__", "assets", "results", "tests", "docs", "dataset/hotpotqa*"]

image = (
    modal.Image.from_registry(os.environ.get("SGLANG_IMAGE", "lmsysorg/sglang:gemma4-mtp"), add_python=None)
    .entrypoint([])
    .apt_install("patch", "gcc", "libc6-dev")
    .pip_install("uv")
    # Same conservative fused-MoE Triton config jevlike ships for the L40S (modal_sglang.py).
    .run_commands(
        "python3 - <<'PY'\n"
        "import json, os\n"
        "dirs=['/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1',"
        "      '/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0']\n"
        "small={'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':1,'num_warps':4,'num_stages':3}\n"
        "big={'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8,'num_warps':4,'num_stages':3}\n"
        "cfg={str(m):(small if m<=32 else big) for m in [1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096]}\n"
        "for d in dirs:\n"
        "  os.makedirs(d, exist_ok=True)\n"
        "  for n in ['E=128,N=704,device_name=NVIDIA_L40S,dtype=fp8_w8a8,per_channel_quant=True.json','E=128,N=704,device_name=NVIDIA_L40S,dtype=fp8_w8a8,per_channel_quant=True_down.json']:\n"
        "    json.dump(cfg, open(os.path.join(d,n),'w'), indent=1)\n"
        "print('wrote moe configs')\n"
        "PY"
    )
    .env({"CC": "gcc", "UV_LINK_MODE": "copy"})
    .add_local_dir("vendor/SAG-Benchmark/sag2", "/opt/upstream/sag2", copy=True, ignore=UPSTREAM_IGNORE)
    .add_local_dir("patches", "/opt/patches", copy=True)
    .run_commands(
        "cd /opt/upstream && patch -p1 < /opt/patches/0001-jevlike-backend.patch",
        "cd /opt/upstream/sag2 && uv sync --frozen && uv pip install tokenizers && uv pip freeze > /opt/upstream/freeze.txt",
        "cd /opt/upstream/sag2 && .venv/bin/python -W ignore scripts/run_sag2_benchmark.py --index-only "
        "--datasets musique 2wikimultihopqa --output-dir /tmp/index-only",
    )
    .add_local_dir("adapters", "/opt/adapters")
    # S4: MultiHop-RAG in the HippoRAG-style layout (tools/convert_multihop_rag.py writes it)
    .add_local_dir("build/mhr_upstream", "/opt/datasets/multihop-rag")
)


def start_server(port=30000, ctx=16384, max_running=64, extra=()):
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", f"/hf/{FP8_REPO}", "--context-length", str(ctx),
           "--max-running-requests", str(max_running), "--mem-fraction-static", "0.8", "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "warning", *extra]
    print("starting:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    for _ in range(1800):
        if proc.poll() is not None:
            raise RuntimeError(f"sglang exited early {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"sglang healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, time.time() - t0
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("sglang not healthy")


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, volumes={"/hf": weights, "/results": results},
              timeout=60 * 60 * 4)
def run(name: str, dataset: str = "musique", limit: int = 10, offset: int = 0, expansions: int = 2,
        concurrency: int = 16, batch: int = 64, max_running: int = 64, ctx: int = 16384, server_extra: str = "",
        dataset_dir: str = ""):
    t_container = time.time()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    ver = subprocess.run(["python3", "-c", "import sglang; print(sglang.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    proc, t_start = start_server(ctx=ctx, max_running=max_running, extra=tuple(server_extra.split()))
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    out = f"/results/{name}"
    os.makedirs(out, exist_ok=True)
    env = {"name": name, "gpu": smi, "sglang_version": ver, "server_start_s": round(t_start, 1),
           "gpu_mem_after_load": mem, "repo": FP8_REPO, "rev": FP8_REV, "context_length": ctx,
           "max_running_requests": max_running, "server_extra": server_extra,
           "args": {"dataset": dataset, "limit": limit, "offset": offset, "expansions": expansions,
                    "concurrency": concurrency, "batch": batch},
           "freeze": open("/opt/upstream/freeze.txt").read().split()}
    print(json.dumps({k: v for k, v in env.items() if k != "freeze"}), flush=True)
    try:
        # dataset may be "musique:300,2wikimultihopqa:100" to run several question sets on one warm server
        specs = [(part.split(":")[0], int(part.split(":")[1]) if ":" in part else limit) for part in dataset.split(",")]
        child_env = dict(os.environ, PYTHONPATH="/opt/adapters", JEVLIKE_URL="http://127.0.0.1:30000",
                         JEVLIKE_CONTEXT=str(ctx))
        runs, code = [], 0
        for name_, limit_ in specs:
            suite = f"{out}/run" if len(specs) == 1 else f"{out}/run-{name_}"
            log = f"{out}/judgments.jsonl" if len(specs) == 1 else f"{out}/judgments-{name_}.jsonl"
            cmd = [".venv/bin/python", "-W", "ignore", "scripts/run_sag2_benchmark.py", "--mode", "local",
                   "--reranker", "jevlike", "--model-path", f"/hf/{FP8_REPO}", "--dataset", name_,
                   "--offset", str(offset), "--limit", str(limit_), "--expansions", str(expansions),
                   "--concurrency", str(concurrency), "--batch-size", str(batch), "--timeout", "1800"]
            if dataset_dir:
                cmd += ["--dataset-dir", dataset_dir, "--cache-dir", f"/tmp/cache-{name}"]
            cmd += ["--resume", suite] if os.path.isdir(suite) else ["--output-dir", suite]
            t0 = time.time()
            rc = subprocess.run(cmd, cwd="/opt/upstream/sag2", env=dict(child_env, JEVLIKE_LOG=log)).returncode
            code = code or rc
            metrics_path = f"{suite}/{name_}/metrics.json"
            metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
            runs.append({"dataset": name_, "limit": limit_, "returncode": rc, "bench_wall_s": round(time.time() - t0, 1),
                         "metrics": metrics and {k: metrics[k] for k in ("question_count", "metrics", "bm25_metrics",
                                                                         "all_evidence_metrics", "requests", "input_tokens")}})
            results.commit()
        total = time.time() - t_container
        env.update(returncode=code, runs=[{k: v for k, v in r.items() if k != "metrics"} for r in runs],
                   bench_wall_s=round(sum(r["bench_wall_s"] for r in runs), 1), container_wall_s=round(total, 1),
                   gpu_usd_per_hour=GPU_USD_PER_HOUR.get(GPU), est_usd=round(total / 3600 * GPU_USD_PER_HOUR.get(GPU, 0), 3))
        json.dump(env, open(f"{out}/_env.json", "w"), indent=1)
        return {"env": {k: v for k, v in env.items() if k != "freeze"}, "runs": runs}
    finally:
        proc.terminate()
        results.commit()


@app.local_entrypoint()
def main(name: str, dataset: str = "musique", limit: int = 10, offset: int = 0, expansions: int = 2,
         concurrency: int = 16, batch: int = 64, max_running: int = 64, context_length: int = 16384,
         server_extra: str = "", dataset_dir: str = ""):
    print(json.dumps(run.remote(name, dataset, limit, offset, expansions, concurrency, batch, max_running, context_length,
                                server_extra, dataset_dir), indent=1, ensure_ascii=False))
