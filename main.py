"""Command-line interface for the RAG PDF summarizer."""

import argparse
import sys

from rag_summarizer import RAGSummarizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-summarizer",
        description="Summarize PDFs and answer questions about them using RAG + Claude.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_cmd = subparsers.add_parser("summarize", help="Summarize a PDF file")
    summarize_cmd.add_argument("pdf", help="Path to the PDF file")
    summarize_cmd.add_argument(
        "--top-k", type=int, default=10, help="Chunks to retrieve for the summary"
    )
    summarize_cmd.add_argument(
        "--max-tokens", type=int, default=2048, help="Max tokens for the response"
    )

    ask_cmd = subparsers.add_parser("ask", help="Ask a question about a PDF file")
    ask_cmd.add_argument("pdf", help="Path to the PDF file")
    ask_cmd.add_argument("question", help="Question to ask")
    ask_cmd.add_argument(
        "--top-k", type=int, default=5, help="Chunks to retrieve as context"
    )
    ask_cmd.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens for the response"
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summarizer = RAGSummarizer()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Loading PDF: {args.pdf}", file=sys.stderr)
    try:
        n_chunks = summarizer.load_pdf(args.pdf)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Indexed {n_chunks} chunks.", file=sys.stderr)

    if args.command == "summarize":
        result = summarizer.summarize(top_k=args.top_k, max_tokens=args.max_tokens)
        print(result)
    elif args.command == "ask":
        result = summarizer.ask(
            args.question, top_k=args.top_k, max_tokens=args.max_tokens
        )
        print(result)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())