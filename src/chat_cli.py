#!/usr/bin/env python3
"""
CLI tool for conversational interface.

Usage:
    # Interactive chat mode
    python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json
    
    # Single question
    python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json \
        --question "Why was Option C recommended?"
"""
from __future__ import annotations

import argparse
import sys

from src.chat_interface import chat_interactive, query_single


def main():
    parser = argparse.ArgumentParser(
        description="Chat interface for querying strategic plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat
  python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json
  
  # Single question
  python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json \\
      --question "Why was Option C recommended?"
        """
    )
    
    parser.add_argument(
        "--pack",
        type=str,
        required=True,
        help="Path to steerco_pack.json file",
    )
    
    parser.add_argument(
        "--plan",
        type=str,
        default="data/plan.yaml",
        help="Path to plan.yaml",
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to ask (non-interactive mode)",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini model to use",
    )
    
    args = parser.parse_args()
    
    if args.question:
        # Single question mode
        answer = query_single(
            question=args.question,
            pack_path=args.pack,
            plan_path=args.plan,
            api_key=args.api_key,
        )
        print(answer)
    else:
        # Interactive mode
        chat_interactive(
            pack_path=args.pack,
            plan_path=args.plan,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
