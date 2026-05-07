from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config.settings import Settings
from app.module4.dependencies import (
    get_chat_model_config,
    get_required_api_key_name,
    has_model_credentials,
    prepare_environment,
)
from app.module4.schemas import (
    DEFAULT_AUDIENCE,
    DEFAULT_MAX_SECTIONS,
    DEFAULT_TOPIC,
    MAX_SECTIONS_LIMIT,
    ChatModelConfig,
)
from app.module4.services.graph_service import Module4GraphExecutionError, run_brief


def section_count(value: str) -> int:
    """Parse a bounded section count argument.

    Args:
        value: Raw command-line value.

    Returns:
        Parsed section count.

    Raises:
        argparse.ArgumentTypeError: If the count is outside the supported range.
    """
    parsed = int(value)
    if parsed < 1 or parsed > MAX_SECTIONS_LIMIT:
        raise argparse.ArgumentTypeError(
            f"value must be between 1 and {MAX_SECTIONS_LIMIT}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the module 4 CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the module 4 LangGraph research brief generator."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f"Research topic to analyze. Default: {DEFAULT_TOPIC}",
    )
    parser.add_argument(
        "--audience",
        default=DEFAULT_AUDIENCE,
        help="Who the brief is for.",
    )
    parser.add_argument(
        "--sections",
        type=section_count,
        default=DEFAULT_MAX_SECTIONS,
        help=f"How many sections to plan. Default: {DEFAULT_MAX_SECTIONS}",
    )
    parser.add_argument(
        "--no-wikipedia",
        action="store_true",
        help="Skip Wikipedia retrieval.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Skip Tavily web retrieval.",
    )
    parser.add_argument(
        "--output",
        help="Optional markdown output path.",
    )
    parser.add_argument(
        "--model",
        help="LangChain chat model name. Defaults to LANGCHAIN_CHAT_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--model-provider",
        help="LangChain model provider. Defaults to LANGCHAIN_MODEL_PROVIDER or openai.",
    )
    return parser


def resolve_model_config(
    *,
    parser: argparse.ArgumentParser,
    settings: Settings,
    model: str | None,
    model_provider: str | None,
) -> ChatModelConfig:
    """Resolve CLI model configuration or stop with a parser error.

    Args:
        parser: CLI parser used to render configuration errors.
        settings: Runtime settings.
        model: Optional explicit model name.
        model_provider: Optional explicit model provider.

    Returns:
        Resolved model configuration.
    """
    try:
        model_config = get_chat_model_config(
            model=model,
            model_provider=model_provider,
            settings=settings,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not has_model_credentials(model_config.model_provider, settings=settings):
        api_key_name = get_required_api_key_name(model_config.model_provider)
        parser.error(
            f"{api_key_name} is not set for provider '{model_config.model_provider}'. "
            "Add it to `.env` or export it in your shell."
        )

    return model_config


def write_output(path: str, report: str) -> None:
    """Write a markdown report to disk.

    Args:
        path: Output file path.
        report: Markdown report body.
    """
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> int:
    """Run module 4 from the command line.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    settings = prepare_environment()
    model_config = resolve_model_config(
        parser=parser,
        settings=settings,
        model=args.model,
        model_provider=args.model_provider,
    )

    try:
        result = run_brief(
            topic=args.topic,
            audience=args.audience,
            max_sections=args.sections,
            include_wikipedia=not args.no_wikipedia,
            include_web=not args.no_web,
            model=model_config.model,
            model_provider=model_config.model_provider,
            settings=settings,
        )
    except Module4GraphExecutionError as exc:
        parser.exit(
            1,
            "error: brief generation failed while calling the model or retrieval "
            f"providers ({exc.__class__.__name__}).\n",
        )

    report = result["final_report"]
    print(report)

    if args.output:
        write_output(args.output, report)
        print(f"\nSaved report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
