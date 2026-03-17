"""GPU selection and CUDA_VISIBLE_DEVICES management.

Called before each training/evaluation/prediction run to let the user
confirm or change which GPUs to use.
"""

import os
import subprocess


GPU_CONFIG_PATH = "/Tmp/hassicir/ai-research/gpu.conf"


def detect_gpus() -> list[str]:
    """Return a list of GPU info lines via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


def load_gpu_config() -> str | None:
    """Load the saved CUDA_VISIBLE_DEVICES value from the global config."""
    if os.path.exists(GPU_CONFIG_PATH):
        with open(GPU_CONFIG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("CUDA_VISIBLE_DEVICES="):
                    return line.split("=", 1)[1]
    return None


def save_gpu_config(value: str):
    """Save CUDA_VISIBLE_DEVICES to the global config file."""
    os.makedirs(os.path.dirname(GPU_CONFIG_PATH), exist_ok=True)
    with open(GPU_CONFIG_PATH, "w") as f:
        f.write(f"CUDA_VISIBLE_DEVICES={value}\n")


def setup_gpu(no_gpu: bool = False, gpus: str | None = None) -> str | None:
    """Set up CUDA_VISIBLE_DEVICES before a run.

    Args:
        no_gpu:  If True, skip the prompt and use all available GPUs.
        gpus:    If provided, use these GPU ids directly (no prompt).

    Returns:
        The CUDA_VISIBLE_DEVICES value that was set, or None if unrestricted.
    """
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        save_gpu_config(gpus)
        print(f"[GPU] CUDA_VISIBLE_DEVICES = {gpus}")
        return gpus

    if no_gpu:
        # Remove any restriction -- use all GPUs
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print("[GPU] Using all available GPUs (no restriction)")
        return None

    # Interactive prompt
    gpu_lines = detect_gpus()
    current = load_gpu_config()

    if gpu_lines:
        print("\n  Available GPUs:")
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            idx, name = parts[0], parts[1]
            mem_total = parts[2] if len(parts) > 2 else "?"
            mem_free = parts[3] if len(parts) > 3 else "?"
            print(f"    [{idx}] {name}  ({mem_free}/{mem_total} MiB free)")
    else:
        print("\n  No NVIDIA GPUs detected.")

    prompt = "  Select GPUs (comma-separated, e.g. 0,1)"
    if current is not None:
        prompt += f" [current: {current}]"
    prompt += ", or Enter for all: "

    value = input(prompt).strip()

    if not value and current is not None:
        # Keep current selection
        os.environ["CUDA_VISIBLE_DEVICES"] = current
        print(f"[GPU] CUDA_VISIBLE_DEVICES = {current}")
        return current
    elif value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        save_gpu_config(value)
        print(f"[GPU] CUDA_VISIBLE_DEVICES = {value}")
        return value
    else:
        # Empty input, no previous config -- use all
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print("[GPU] Using all available GPUs (no restriction)")
        return None


if __name__ == "__main__":
    setup_gpu()
