"""Convert a Hugging Face model directory to GGUF and quantize it with llama.cpp.

    python scripts/export_gguf.py --hf build/finetuned/merged --name finetuned \
        --llama-cpp llama.cpp --types F16 Q8_0 Q4_K_M

Writes build/gguf/<name>-<type>.gguf for each type.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "gguf")
    ap.add_argument("--types", nargs="+", default=["F16", "Q8_0", "Q4_K_M"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    f16 = args.out / f"{args.name}-F16.gguf"
    subprocess.run(
        [sys.executable, str(args.llama_cpp / "convert_hf_to_gguf.py"), str(args.hf),
         "--outtype", "f16", "--outfile", str(f16)],
        check=True,
    )  # fmt: skip
    quantize = args.llama_cpp / "build" / "bin" / "llama-quantize"
    for t in args.types:
        if t == "F16":
            continue
        subprocess.run([str(quantize), str(f16), str(args.out / f"{args.name}-{t}.gguf"), t], check=True)
    if "F16" not in args.types:
        f16.unlink()
    for p in sorted(args.out.glob(f"{args.name}-*.gguf")):
        print(f"{p.name} {p.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
