"""CLI entry point for AI Investment Risk Radar pipeline."""

import argparse

from ai_risk_radar.config import load_settings
from ai_risk_radar.pipeline import run_final_test, run_latest, run_research


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Investment Risk Radar CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Research Command
    res_parser = subparsers.add_parser("research", help="Fit models and select strategy on validation partition")
    res_parser.add_argument("--config", default="config.json", help="Path to configuration file")
    res_parser.add_argument("--data-mode", choices=["refresh", "cache"], default="cache", help="Data mode: refresh or cache")

    # Final Test Command
    test_parser = subparsers.add_parser("final-test", help="Evaluate frozen release on untouched holdout test partition")
    test_parser.add_argument("--release", default="artifacts/release.json", help="Path to release manifest")
    test_parser.add_argument("--config", default=None, help="Optional path to configuration file")

    # Latest Inference Command
    lat_parser = subparsers.add_parser("latest", help="Generate real-time risk estimate using frozen model")
    lat_parser.add_argument("--release", default="artifacts/release.json", help="Path to release manifest")
    lat_parser.add_argument("--config", default=None, help="Optional path to configuration file")
    lat_parser.add_argument("--data-mode", choices=["refresh", "cache"], default="cache", help="Data mode: refresh or cache")

    args = parser.parse_args()

    if args.command == "research":
        settings = load_settings(args.config)
        out = run_research(settings, data_mode=args.data_mode)
        print(f"Research completed successfully. Report generated at: {out}")
    elif args.command == "final-test":
        out = run_final_test(args.release, config_path=args.config)
        print(f"Final holdout test evaluated. Report generated at: {out}")
    elif args.command == "latest":
        out = run_latest(args.release, data_mode=args.data_mode, config_path=args.config)
        print(f"Latest risk report published at: {out}")


if __name__ == "__main__":
    main()
