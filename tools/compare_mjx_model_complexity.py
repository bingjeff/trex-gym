#!/usr/bin/env python3
"""Compares full and simplified MJCF model complexity."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mjx_model_simplification
from tools import urdf_parsing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full and simplified MJX training model complexity."
    )
    parser.add_argument("urdf_path", type=Path)
    parser.add_argument(
        "--position-kp",
        type=float,
        default=mjx_model_simplification.DEFAULT_POSITION_KP,
        help="Proportional gain for generated position actuators.",
    )
    args = parser.parse_args()

    urdf = urdf_parsing.Urdf.from_element(
        urdf_parsing.read_root_node_from_urdf(str(args.urdf_path))
    )
    comparison = mjx_model_simplification.compare_full_and_simplified(
        urdf,
        asset_dir=args.urdf_path.parent,
        position_kp=args.position_kp,
    )
    print(mjx_model_simplification.comparison_markdown(comparison))


if __name__ == "__main__":
    main()
