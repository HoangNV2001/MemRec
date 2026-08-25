#!/usr/bin/env python3
"""Validate the active Azure LLM configuration without exposing its API key.

Use --request only after .env contains a real LLM__API_KEY:

    python scripts/check_llm_config.py
    python scripts/check_llm_config.py --request
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.models.llm_client import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Azure LLM configuration")
    parser.add_argument("--request", action="store_true", help="make one small Chat Completions request")
    args = parser.parse_args()

    required = ("LLM__BASE_URL", "LLM__API_KEY", "LLM__API_VERSION", "LLM__MODEL_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Missing active Azure configuration in .env: "
            + ", ".join(missing)
            + ". Copy the names from .env.example and set a real LLM__API_KEY."
        )

    client = LLMClient(provider_name="azure_openai")
    print("Azure LLM configuration")
    print(f"  endpoint: {client.api_endpoint}")
    print(f"  api version: {client.api_version}")
    print(f"  configured model: {client.model}")
    print(f"  Azure deployment sent to API: {client.request_model}")

    if args.request:
        response = client.generate(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0.0,
            max_retries=1,
        )
        print(f"  smoke response: {response!r}")


if __name__ == "__main__":
    main()
