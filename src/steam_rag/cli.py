from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

from .corpus import OnDemandCorpusManager
from .index import VectorIndex, build_index
from .markdown_corpus import load_documents
from .openai_adapter import OpenAIAnswerGenerator, OpenAIEmbedder, load_env_file
from .pipeline import RAGPipeline
from .rerank import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from .steam import SteamAPIClient, SteamGame, save_catalog


DEFAULT_INDEX = Path("data/vectorstore/steam_rag_timeaware_playstyle.json")
DEFAULT_DOCS = Path("data/docs_timeaware_playstyle")
DEFAULT_RAW = Path("data/raw/on_demand")
DEFAULT_CATALOG = Path("data/steam_catalog.json")


def _add_corpus_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    command.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    command.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    command.add_argument("--max-age-hours", type=float, default=24.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steam game hybrid time-aware RAG pipeline")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Parse Markdown, embed chunks, and persist an index")
    build.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--embedding-model", default="text-embedding-3-small")
    build.add_argument("--chunk-size", type=int, default=1000)
    build.add_argument("--chunk-overlap", type=int, default=200)
    build.add_argument("--batch-size", type=int, default=64)

    inspect = subparsers.add_parser("inspect", help="Show safe index metadata without API calls")
    inspect.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    sync_catalog = subparsers.add_parser(
        "sync-catalog", help="Synchronize the Steam app catalog using STEAM_WEB_API_KEY"
    )
    sync_catalog.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)

    collect = subparsers.add_parser(
        "collect",
        help="Collect one Steam game into combined Markdown and upsert the vector store",
    )
    collect.add_argument("--appid", type=int, required=True)
    collect.add_argument("--name", default="Steam Game")
    collect.add_argument("--game-key", default="")
    collect.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    collect.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    collect.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    collect.add_argument("--embedding-model", default="text-embedding-3-small")
    collect.add_argument("--max-reviews", type=int, default=50)
    collect.add_argument("--news-count", type=int, default=20)

    ensure = subparsers.add_parser(
        "ensure", help="Resolve a question's game and collect/index it when missing or stale"
    )
    ensure.add_argument("question")
    ensure.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ensure.add_argument("--embedding-model", default="text-embedding-3-small")
    ensure.add_argument("--force", action="store_true")
    _add_corpus_arguments(ensure)

    for name, help_text in (
        ("search", "Retrieve evidence without answer generation"),
        ("ask", "Retrieve evidence and generate a grounded answer"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("question")
        command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        command.add_argument("--embedding-model", default="text-embedding-3-small")
        command.add_argument(
            "--reranker-model",
            default="",
            help=f"Optional cross-encoder reranker model, e.g. {DEFAULT_RERANKER_MODEL}",
        )
        command.add_argument("--rerank-candidates", type=int, default=24)
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument(
            "--auto-collect",
            action="store_true",
            help="Collect and incrementally index a missing/stale queried game before retrieval",
        )
        _add_corpus_arguments(command)
        if name == "ask":
            command.add_argument("--answer-model", default="gpt-5-mini")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        index = VectorIndex.load(args.index)
        sections: dict[str, int] = {}
        for document in index.documents:
            section = str(document.metadata.get("section", "unknown"))
            sections[section] = sections.get(section, 0) + 1
        print(
            json.dumps(
                {
                    "index": str(args.index),
                    "embedding_model": index.embedding_model,
                    "document_count": len(index.documents),
                    "sections": sections,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    load_env_file(args.env_file)
    if args.command == "sync-catalog":
        api_key = os.getenv("STEAM_WEB_API_KEY", "")
        apps = SteamAPIClient().fetch_catalog(api_key)
        save_catalog(args.catalog, apps)
        print(json.dumps({"catalog": str(args.catalog), "app_count": len(apps)}, indent=2))
        return 0

    if args.command == "collect":
        embedder = OpenAIEmbedder(args.embedding_model)
        update = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=DEFAULT_CATALOG,
            docs_dir=args.docs,
            raw_dir=args.raw,
            index_path=args.index,
            max_reviews=args.max_reviews,
            news_count=args.news_count,
        ).ensure_game(
            SteamGame(args.appid, args.name, args.game_key),
            embedder,
            force=True,
        )
        print(
            json.dumps(
                {
                    "markdown": str(update.markdown_path),
                    "vectorstore": str(args.index),
                    "appid": update.game.appid,
                    "name": update.game.name,
                    "collected": update.collected,
                    "indexed": update.indexed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    embedder = OpenAIEmbedder(args.embedding_model)
    if args.command == "build":
        documents = load_documents(
            args.docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        index = build_index(documents, embedder, batch_size=args.batch_size)
        index.save(args.index)
        print(json.dumps({"index": str(args.index), "document_count": len(documents)}, indent=2))
        return 0


    def corpus_manager() -> OnDemandCorpusManager:
        return OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=args.catalog,
            docs_dir=args.docs,
            raw_dir=args.raw,
            index_path=args.index,
            max_age=timedelta(hours=args.max_age_hours),
        )

    if args.command == "ensure":
        update = corpus_manager().ensure_question(args.question, embedder, force=args.force)
        print(
            json.dumps(
                {
                    "appid": update.game.appid,
                    "name": update.game.name,
                    "markdown": str(update.markdown_path),
                    "collected": update.collected,
                    "indexed": update.indexed,
                    "reason": update.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.auto_collect:
        corpus_manager().ensure_question(args.question, embedder)

    generator = OpenAIAnswerGenerator(args.answer_model) if args.command == "ask" else None
    reranker = CrossEncoderReranker(args.reranker_model) if args.reranker_model else None
    pipeline = RAGPipeline.from_path(
        args.index,
        embedder,
        generator,
        reranker=reranker,
        rerank_candidates=args.rerank_candidates,
    )
    if args.command == "search":
        results = pipeline.search(args.question, k=args.top_k)
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(pipeline.ask(args.question, k=args.top_k).to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
