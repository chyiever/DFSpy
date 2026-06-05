"""Execute notebooks cell by cell with jupyter_client and save outputs via UTF-8 JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

from jupyter_client import KernelManager


ROOT = Path(r"E:\codes\DFSpy")
NOTEBOOK_DIR = ROOT / "FJ-QED" / "FJpy-win" / "examples"
IPYTHON_DIR = ROOT / ".ipython_temp"


def collect_messages(kernel_client, msg_id: str, timeout: int = 1800) -> list[dict]:
    outputs: list[dict] = []
    while True:
        msg = kernel_client.get_iopub_msg(timeout=timeout)
        parent = msg.get("parent_header", {})
        if parent.get("msg_id") != msg_id:
            continue
        msg_type = msg["msg_type"]
        content = msg["content"]
        if msg_type == "status" and content.get("execution_state") == "idle":
            break
        if msg_type == "stream":
            outputs.append(
                {
                    "output_type": "stream",
                    "name": content.get("name", "stdout"),
                    "text": content.get("text", ""),
                }
            )
        elif msg_type in ("display_data", "execute_result"):
            item = {
                "output_type": msg_type,
                "data": content.get("data", {}),
                "metadata": content.get("metadata", {}),
            }
            if msg_type == "execute_result":
                item["execution_count"] = content.get("execution_count")
            outputs.append(item)
        elif msg_type == "error":
            outputs.append(
                {
                    "output_type": "error",
                    "ename": content.get("ename", ""),
                    "evalue": content.get("evalue", ""),
                    "traceback": content.get("traceback", []),
                }
            )
    return outputs


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(path.parent))
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=60)
        setup_id = kc.execute("get_ipython().run_line_magic('matplotlib', 'inline')")
        collect_messages(kc, setup_id, timeout=60)
        exec_count = 1
        for idx, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            print(f"{path.name} RUN cell {idx}", flush=True)
            msg_id = kc.execute("".join(cell.get("source", [])))
            outputs = collect_messages(kc, msg_id, timeout=2400)
            cell["execution_count"] = exec_count
            cell["outputs"] = outputs
            if any(output.get("output_type") == "error" for output in outputs):
                raise RuntimeError(f"Execution error in {path.name} cell {idx}")
            exec_count += 1
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    text = json.dumps(notebook, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    print(f"EXECUTED {path.name}", flush=True)


def main() -> None:
    IPYTHON_DIR.mkdir(exist_ok=True)
    os.environ["IPYTHONDIR"] = str(IPYTHON_DIR)
    for name in ["example_CC.ipynb", "example_noise.ipynb", "example_EQ.ipynb"]:
        execute_notebook(NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
