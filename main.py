"""Mission router entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone AI mission router")
    parser.add_argument("--mission", choices=("facility", "mapping", "obstacle"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate routing without running a pipeline")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Dry run OK: {args.mission} mission selected")
        return

    if args.mission == "facility":
        from mission.facility_pipeline import run
    elif args.mission == "mapping":
        from mission.map_pipeline import run
    else:
        from mission.obstacle_pipeline import run
    run()


if __name__ == "__main__":
    main()
