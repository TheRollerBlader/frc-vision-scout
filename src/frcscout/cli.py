import argparse

from frcscout import __version__

STAGES = ["ingest", "render", "train", "track", "report"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="frcscout", description="FRC vision scouting pipeline")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("stage", nargs="?", choices=STAGES)
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
    else:
        print(f"[{args.stage}] not implemented yet")


if __name__ == "__main__":
    main()
