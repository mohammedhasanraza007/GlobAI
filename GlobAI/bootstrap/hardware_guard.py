"""
bootstrap/hardware_guard.py
---------------------------
RAM/GPU visibility checks for local deployment.
"""

from __future__ import annotations

import logging
import sys

import psutil

logger = logging.getLogger(__name__)

MIN_RAM_GB = 12.0
MIN_AVAILABLE_RAM_GB = 3.0


def check_hardware() -> dict:
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024**3)
    available_ram_gb = vm.available / (1024**3)
    info = {
        "ram_total_gb": round(total_ram_gb, 2),
        "ram_available_gb": round(available_ram_gb, 2),
        "gpu_available": False,
        "gpu_name": "N/A",
        "vram_total_mb": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_available"] = True
            info["gpu_name"] = props.name
            info["vram_total_mb"] = props.total_memory // (1024 * 1024)
    except Exception as exc:
        logger.debug("CUDA probe failed: %s", exc)

    if not info["gpu_available"]:
        try:
            import torch_directml  # type: ignore

            if not hasattr(torch_directml, "is_available") or torch_directml.is_available():
                info["gpu_available"] = True
                info["gpu_name"] = f"DirectML:{torch_directml.device_name(0)}"
        except Exception:
            pass

    _report(info)

    if total_ram_gb < MIN_RAM_GB:
        print(
            f"\n[HW_GUARD] ABORT - insufficient RAM.\n"
            f"           Required: {MIN_RAM_GB:.0f} GB | Detected: {total_ram_gb:.1f} GB\n"
        )
        sys.exit(1)

    if available_ram_gb < MIN_AVAILABLE_RAM_GB:
        print(
            f"\n[HW_GUARD] WARNING - only {available_ram_gb:.1f} GB RAM is available. "
            "Close other applications before loading the LLM.\n"
        )

    return info


def _report(info: dict) -> None:
    lines = [
        "[HW_GUARD] Hardware Report:",
        f"  RAM Total    : {info['ram_total_gb']} GB",
        f"  RAM Available: {info['ram_available_gb']} GB",
        f"  GPU          : {('YES - ' + info['gpu_name']) if info['gpu_available'] else 'None (CPU mode)'}",
    ]
    if info["vram_total_mb"]:
        lines.append(f"  VRAM Total   : {info['vram_total_mb']} MB")
    for line in lines:
        print(line)
        logger.info(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_hardware()
    print("\n[HW_GUARD] Hardware check passed.")
