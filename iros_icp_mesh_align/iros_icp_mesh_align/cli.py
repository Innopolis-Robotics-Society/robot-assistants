from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_alignment_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh(mesh)->PCD->RANSAC+ICP->Heatmap (no logging)")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--auto_voxel", action="store_true", help="Enable auto voxel (recommended)")
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--voxel_ratio", type=float, default=0.005)
    ap.add_argument("--voxel_min", type=float, default=1e-3)

    ap.add_argument("--n_coarse", type=int, default=30000)
    ap.add_argument("--n_fine", type=int, default=200000)
    ap.add_argument("--sample_method", default="poisson", choices=["poisson", "uniform"])
    ap.add_argument("--icp_max_iter", type=int, default=60)
    ap.add_argument("--outliers", action="store_true")

    ap.add_argument("--dist_thresh", type=float, default=2.0)
    ap.add_argument("--heat_vmin", type=float, default=0.0)
    ap.add_argument("--heat_vmax", type=float, default=5.0)

    args = ap.parse_args()

    try:
        out = run_alignment_pipeline(
            ref_path=args.ref,
            scan_path=args.scan,
            out_dir=args.out,
            auto_voxel=bool(args.auto_voxel),
            voxel=args.voxel,
            voxel_ratio=args.voxel_ratio,
            voxel_min=args.voxel_min,
            n_coarse=args.n_coarse,
            n_fine=args.n_fine,
            sample_method=args.sample_method,
            icp_max_iter=args.icp_max_iter,
            remove_outliers=bool(args.outliers),
            dist_thresh=args.dist_thresh,
            heat_vmin=args.heat_vmin,
            heat_vmax=args.heat_vmax,
            log=None,
            warn=None,
        )
        print(json.dumps(out.report, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
