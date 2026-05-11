#!/usr/bin/env python3
"""Converts a URDF file into a simplified MJCF XML for MJX training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import mjx_model_simplification
from tools import mujoco_parsing
from tools import urdf_parsing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a URDF model into a simplified MJX training MJCF."
    )
    parser.add_argument("urdf_path", type=Path)
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Output XML path. Prints to stdout when omitted.",
    )
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
    mjcf = mjx_model_simplification.urdf_to_mjx_mujoco(
        urdf, position_kp=args.position_kp
    )
    xml = mujoco_parsing.to_string(mjcf)
    if args.output_path:
        args.output_path.write_text(xml)
    else:
        print(xml)


if __name__ == "__main__":
    main()
