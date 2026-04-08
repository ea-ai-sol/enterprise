import os
import sys
import argparse

try:
    from openai import OpenAI
except Exception:
    print(
        "Import error: install/upgrade the OpenAI SDK in this interpreter with "
        "`python3 -m pip install -U openai`",
        file=sys.stderr,
    )
    raise


def load_api_key_from_dotenv(dotenv_path: str = ".env") -> str:
    """Read OPENAI_API_KEY from a local .env file if present."""
    if not os.path.exists(dotenv_path):
        return ""

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    return value.strip().strip("'").strip('"')
    except OSError:
        return ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple OpenAI hello-world caller")
    parser.add_argument("--api-key", default="", help="OpenAI API key (optional)")
    args = parser.parse_args()

    # Priority: CLI flag -> environment variable -> .env file
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or load_api_key_from_dotenv()
    if not api_key:
        print(
            "Error: no API key found. Use --api-key, set OPENAI_API_KEY, "
            "or add OPENAI_API_KEY=... to .env",
            file=sys.stderr,
        )
        return 1

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input="Say hello world with 3 blank lines before and after.",
        )
    except Exception as err:
        print(f"OpenAI request failed: {err}", file=sys.stderr)
        return 1

    print(response.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
