from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from steam_rag.application.rag_pipeline import RAGPipeline
from steam_rag.evaluation_tools.benchmark import (
    STAGE4_STRATEGIES,
    SUPPORTED_STRATEGIES,
    ServiceConversationBenchmarkRunner,
    Stage4BenchmarkRunner,
    load_conversation_golden_set,
    load_golden_set,
    save_benchmark,
    save_conversation_benchmark,
)
from steam_rag.evaluation_tools.semantic_quality import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RAGAS_MODEL,
)
from steam_rag.external_apis.openai_client import OpenAIAnswerGenerator, OpenAIEmbedder, load_env_file
from steam_rag.game_analysis.time_aware import run_time_analysis_and_index
from steam_rag.game_recommendation.candidate_service import DynamicRecommendationService
from steam_rag.game_recommendation.profile_builder import collect_recommendation_profile
from steam_rag.game_recommendation.profile_store import SteamProfileStore
from steam_rag.game_recommendation.query_parser import (
    OpenAIRecommendationQueryStructurer,
    RecommendationProfileIndex,
    parse_recommendation_query,
)
from steam_rag.rag_search.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from steam_rag.rag_search.vector_store import VectorIndex, build_index
from steam_rag.steam_collection.corpus_manager import OnDemandCorpusManager
from steam_rag.steam_collection.markdown_documents import load_documents
from steam_rag.steam_collection.steam_client import DEFAULT_COUNTRY, DEFAULT_LANGUAGE, SteamAPIClient, SteamGame, save_catalog


DEFAULT_INDEX = Path("data/chroma/steam_rag_timeaware_playstyle")
DEFAULT_DOCS = Path("data/docs_timeaware_playstyle")
DEFAULT_RAW = Path("data/raw/on_demand")
DEFAULT_CATALOG = Path("data/steam_catalog.json")
DEFAULT_PROFILES = Path("data/game_profiles")
DEFAULT_TIME_ANALYSIS = Path("data/time_analysis")
DEFAULT_SERVICE_DB = Path("data/steam_service.db")
DEFAULT_STAGE4_GOLDEN = Path("data/eval/stage4_golden_set.jsonl")
DEFAULT_STAGE4_DETAILS = Path("data/eval/stage4_benchmark_details.jsonl")
DEFAULT_STAGE4_SUMMARY = Path("data/eval/stage4_benchmark_summary.csv")
DEFAULT_CONVERSATION_GOLDEN = Path("data/eval/conversation_golden_set_v1.jsonl")
DEFAULT_CONVERSATION_DETAILS = Path("data/eval/conversation_benchmark_details.jsonl")
DEFAULT_CONVERSATION_SUMMARY = Path("data/eval/conversation_benchmark_summary.json")
DEFAULT_CONVERSATION_MANIFEST = Path("data/eval/conversation_benchmark_manifest.json")


def _merge_quality_run_summaries(
    prepared_dir: Path,
    current: dict[str, object],
    *,
    invoked_engines: list[str],
) -> dict[str, object]:
    """Keep both engine summaries when RAGAS and Judge are resumed separately."""

    merged = dict(current)
    for engine in ("ragas", "judge"):
        if engine in merged:
            continue
        summary_path = prepared_dir / f"{engine}-summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(summary, dict):
            merged[engine] = summary
    merged["invoked_engines"] = list(invoked_engines)
    merged["engines"] = [
        engine for engine in ("ragas", "judge") if isinstance(merged.get(engine), dict)
    ]
    return merged


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


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

    profiles = subparsers.add_parser(
        "build-profiles",
        help="Build Korean recommendation profiles from the Steam app catalog",
    )
    profiles.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    profiles.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    profiles.add_argument("--limit", type=int, default=100)
    profiles.add_argument("--start-after-appid", type=int, default=0)
    profiles.add_argument("--appid", type=int, action="append", default=[])
    profiles.add_argument("--force", action="store_true")
    profiles.add_argument("--language", default=DEFAULT_LANGUAGE)
    profiles.add_argument("--country", default=DEFAULT_COUNTRY)

    recommend = subparsers.add_parser(
        "recommend",
        help="Structure a recommendation query and select Top 20 candidates / Top 5 detail targets",
    )
    recommend.add_argument("question")
    recommend.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    recommend.add_argument("--query-model", default="gpt-5-mini")
    recommend.add_argument("--rule-only", action="store_true")
    recommend.add_argument("--candidate-limit", type=int, default=20)
    recommend.add_argument("--detail-limit", type=int, default=5)

    sync_registry = subparsers.add_parser(
        "sync-registry",
        help="Import the Steam app catalog and existing core profiles into SQLite",
    )
    sync_registry.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    sync_registry.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    sync_registry.add_argument("--service-db", type=Path, default=DEFAULT_SERVICE_DB)

    service_recommend = subparsers.add_parser(
        "recommend-service",
        help="Dynamically expand missing core profiles and optionally enrich Top 5 details",
    )
    service_recommend.add_argument("question")
    service_recommend.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    service_recommend.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    service_recommend.add_argument("--service-db", type=Path, default=DEFAULT_SERVICE_DB)
    service_recommend.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    service_recommend.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    service_recommend.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    service_recommend.add_argument("--time-analysis", type=Path, default=DEFAULT_TIME_ANALYSIS)
    service_recommend.add_argument("--query-model", default="gpt-5-mini")
    service_recommend.add_argument("--embedding-model", default="text-embedding-3-small")
    service_recommend.add_argument("--rule-only", action="store_true")
    service_recommend.add_argument("--no-expand", action="store_true")
    service_recommend.add_argument("--enrich-details", action="store_true")
    service_recommend.add_argument("--min-candidates", type=int, default=20)
    service_recommend.add_argument("--max-new-profiles", type=int, default=20)
    service_recommend.add_argument("--discovery-per-term", type=int, default=30)
    service_recommend.add_argument("--candidate-limit", type=int, default=20)
    service_recommend.add_argument("--detail-limit", type=int, default=5)

    analyze_update = subparsers.add_parser(
        "analyze-update",
        help="Compare Steam review sentiment and topics before/after a structured patch event",
    )
    analyze_update.add_argument("--appid", type=int, required=True)
    analyze_update.add_argument("--name", default="Steam Game")
    analyze_update.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    analyze_update.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    analyze_update.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    analyze_update.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    analyze_update.add_argument("--output-dir", type=Path, default=DEFAULT_TIME_ANALYSIS)
    analyze_update.add_argument("--embedding-model", default="text-embedding-3-small")
    analyze_update.add_argument("--before-days", type=int, default=30)
    analyze_update.add_argument("--after-days", type=int, default=30)
    analyze_update.add_argument("--max-reviews", type=int, default=5000)
    analyze_update.add_argument("--max-pages", type=int, default=100)
    analyze_update.add_argument("--news-count", type=int, default=100)
    analyze_update.add_argument("--patch-date", default="")
    analyze_update.add_argument("--focus", action="append", default=[])
    analyze_update.add_argument("--language", default=DEFAULT_LANGUAGE)

    evaluate = subparsers.add_parser(
        "evaluate-stage4",
        help="Compare Agentic RAG with and without HyDE on the 50-case Golden Set",
    )
    evaluate.add_argument("--golden-set", type=Path, default=DEFAULT_STAGE4_GOLDEN)
    evaluate.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluate.add_argument("--details-output", type=Path, default=DEFAULT_STAGE4_DETAILS)
    evaluate.add_argument("--summary-output", type=Path, default=DEFAULT_STAGE4_SUMMARY)
    evaluate.add_argument("--embedding-model", default="text-embedding-3-small")
    evaluate.add_argument("--answer-model", default="gpt-5-mini")
    evaluate.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    evaluate.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=list(STAGE4_STRATEGIES),
        help="Defaults to agentic agentic_hyde; earlier retrieval stages remain optional diagnostics",
    )
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--rerank-candidates", type=int, default=24)
    evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--retrieval-only", action="store_true")

    evaluate_conversations = subparsers.add_parser(
        "evaluate-conversations",
        help="Validate or run the 20-case Korean multi-turn service Golden Set",
    )
    evaluate_conversations.add_argument(
        "--golden-set", type=Path, default=DEFAULT_CONVERSATION_GOLDEN
    )
    evaluate_conversations.add_argument(
        "--details-output", type=Path, default=DEFAULT_CONVERSATION_DETAILS
    )
    evaluate_conversations.add_argument(
        "--summary-output", type=Path, default=DEFAULT_CONVERSATION_SUMMARY
    )
    evaluate_conversations.add_argument(
        "--manifest-output", type=Path, default=DEFAULT_CONVERSATION_MANIFEST
    )
    evaluate_conversations.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    evaluate_conversations.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    evaluate_conversations.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    evaluate_conversations.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    evaluate_conversations.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    evaluate_conversations.add_argument("--service-db", type=Path, default=DEFAULT_SERVICE_DB)
    evaluate_conversations.add_argument("--time-analysis", type=Path, default=DEFAULT_TIME_ANALYSIS)
    evaluate_conversations.add_argument("--embedding-model", default="text-embedding-3-small")
    evaluate_conversations.add_argument("--answer-model", default="gpt-5-mini")
    evaluate_conversations.add_argument("--top-k", type=int, default=6)
    evaluate_conversations.add_argument("--limit", type=int, default=0)
    evaluate_conversations.add_argument("--run-id", default="")
    evaluate_conversations.add_argument("--run-label", default="")
    evaluate_conversations.add_argument(
        "--cache-state",
        choices=("unknown", "cold", "warm", "mixed"),
        default="unknown",
        help="Operator label only; caches are never deleted automatically",
    )
    evaluate_conversations.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the Golden Set schema without loading the service or calling external APIs",
    )

    prepare_quality = subparsers.add_parser(
        "prepare-quality-evaluation",
        help="Create fixed RAGAS and LLM Judge inputs from a conversation details JSONL",
    )
    prepare_quality.add_argument("--details", type=Path, required=True)
    prepare_quality.add_argument("--output-dir", type=Path, required=True)

    evaluate_quality = subparsers.add_parser(
        "evaluate-quality",
        help="Run resumable RAGAS and/or LLM Judge evaluation on prepared inputs",
    )
    evaluate_quality.add_argument("--prepared-dir", type=Path, required=True)
    evaluate_quality.add_argument(
        "--engines",
        nargs="+",
        choices=("ragas", "judge"),
        default=("ragas", "judge"),
    )
    evaluate_quality.add_argument("--ragas-model", default=DEFAULT_RAGAS_MODEL)
    evaluate_quality.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    evaluate_quality.add_argument("--embedding-model", default="text-embedding-3-small")
    evaluate_quality.add_argument(
        "--ragas-limit",
        type=int,
        default=0,
        help="Maximum new RAGAS rows in this invocation; 0 evaluates all pending rows",
    )
    evaluate_quality.add_argument(
        "--judge-limit",
        type=int,
        default=0,
        help="Maximum new LLM Judge calls in this invocation; 0 evaluates all pending rows",
    )
    evaluate_quality.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed rows from existing output JSONL files",
    )

    collect = subparsers.add_parser(
        "collect",
        help="Collect one Steam game into combined Markdown and upsert the vector store",
    )
    collect.add_argument("--appid", type=int, required=True)
    collect.add_argument("--name", default="Steam Game")
    collect.add_argument("--game-key", default="")
    collect.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    collect.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    collect.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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

    if args.command == "evaluate-conversations" and args.validate_only:
        cases = load_conversation_golden_set(args.golden_set)
        print(
            json.dumps(
                {
                    "valid": True,
                    "golden_set": str(args.golden_set),
                    "case_count": len(cases),
                    "turn_count": sum(len(case.turns) for case in cases),
                    "categories": sorted({case.category for case in cases}),
                    "external_calls": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "prepare-quality-evaluation":
        from steam_rag.evaluation_tools.semantic_quality import prepare_semantic_evaluation

        summary = prepare_semantic_evaluation(args.details, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    load_env_file(args.env_file)
    if args.command == "evaluate-quality":
        from steam_rag.evaluation_tools.semantic_quality import (
            OpenAIQualityJudge,
            OpenAIRagasEvaluator,
            load_jsonl,
            run_llm_judge,
            run_ragas_evaluation,
        )

        prepared_dir = args.prepared_dir
        invoked_engines = list(args.engines)
        run_summary: dict[str, object] = {
            "prepared_dir": str(prepared_dir),
            "engines": invoked_engines,
            "resume": bool(args.resume),
        }
        if "ragas" in args.engines:
            ragas_rows = load_jsonl(prepared_dir / "ragas-input.jsonl")
            run_summary["ragas"] = run_ragas_evaluation(
                ragas_rows,
                evaluator=OpenAIRagasEvaluator(args.ragas_model, args.embedding_model),
                output_path=prepared_dir / "ragas-results.jsonl",
                summary_path=prepared_dir / "ragas-summary.json",
                limit=max(0, args.ragas_limit),
                resume=bool(args.resume),
            )
        if "judge" in args.engines:
            judge_rows = load_jsonl(prepared_dir / "judge-input.jsonl")
            run_summary["judge"] = run_llm_judge(
                judge_rows,
                judge=OpenAIQualityJudge(args.judge_model),
                output_path=prepared_dir / "judge-results.jsonl",
                summary_path=prepared_dir / "judge-summary.json",
                limit=max(0, args.judge_limit),
                resume=bool(args.resume),
            )
        combined_path = prepared_dir / "quality-run-summary.json"
        run_summary = _merge_quality_run_summaries(
            prepared_dir,
            run_summary,
            invoked_engines=invoked_engines,
        )
        run_summary["summary_output"] = str(combined_path)
        combined_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "sync-catalog":
        api_key = os.getenv("STEAM_WEB_API_KEY", "")
        apps = SteamAPIClient().fetch_catalog(api_key)
        save_catalog(args.catalog, apps)
        print(json.dumps({"catalog": str(args.catalog), "app_count": len(apps)}, indent=2))
        return 0

    if args.command == "build-profiles":
        if not args.catalog.exists():
            api_key = os.getenv("STEAM_WEB_API_KEY", "")
            apps = SteamAPIClient().fetch_catalog(api_key)
            save_catalog(args.catalog, apps)
        payload = json.loads(args.catalog.read_text(encoding="utf-8"))
        apps = payload.get("apps", payload)
        if not isinstance(apps, list):
            raise ValueError(f"Invalid Steam catalog: {args.catalog}")
        requested_appids = set(args.appid)
        selected: list[SteamGame] = []
        for app in apps:
            if not isinstance(app, dict):
                continue
            try:
                appid = int(app.get("appid") or app.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(app.get("name") or "").strip()
            if not name or appid <= args.start_after_appid:
                continue
            if requested_appids and appid not in requested_appids:
                continue
            selected.append(SteamGame(appid, name))
            if args.limit > 0 and len(selected) >= args.limit:
                break

        client = SteamAPIClient()
        completed: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for game in selected:
            existing = next(args.profiles.glob(f"*_{game.appid}.json"), None)
            if existing and not args.force:
                completed.append({"appid": game.appid, "name": game.name, "profile": str(existing), "status": "skipped"})
                continue
            try:
                path = collect_recommendation_profile(
                    client,
                    game,
                    profiles_dir=args.profiles,
                    language=args.language,
                    country=args.country,
                )
                completed.append({"appid": game.appid, "name": game.name, "profile": str(path), "status": "created"})
            except Exception as exc:
                failures.append({"appid": game.appid, "name": game.name, "error": f"{type(exc).__name__}: {exc}"})
        print(
            json.dumps(
                {
                    "catalog": str(args.catalog),
                    "profiles_dir": str(args.profiles),
                    "language": args.language,
                    "country": args.country,
                    "selected_count": len(selected),
                    "completed_count": len(completed),
                    "failure_count": len(failures),
                    "completed": completed,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "recommend":
        query = (
            parse_recommendation_query(args.question)
            if args.rule_only
            else OpenAIRecommendationQueryStructurer(args.query_model).structure(args.question)
        )
        selection = RecommendationProfileIndex.load(args.profiles).search(
            args.question,
            query,
            candidate_limit=args.candidate_limit,
            detail_limit=args.detail_limit,
        )
        print(json.dumps(selection.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "sync-registry":
        store = SteamProfileStore(args.service_db)
        registry_count = store.sync_catalog_file(args.catalog)
        imported_profiles = store.import_profile_directory(args.profiles)
        print(
            json.dumps(
                {
                    "service_db": str(args.service_db),
                    "registry_synced": registry_count,
                    "profiles_imported": imported_profiles,
                    **store.summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "recommend-service":
        store = SteamProfileStore(args.service_db)
        if store.summary()["registry_count"] == 0 and args.catalog.exists():
            store.sync_catalog_file(args.catalog)
        query = (
            parse_recommendation_query(args.question)
            if args.rule_only
            else OpenAIRecommendationQueryStructurer(args.query_model).structure(args.question)
        )
        embedder = OpenAIEmbedder(args.embedding_model) if args.enrich_details else None
        run = DynamicRecommendationService(
            client=SteamAPIClient(),
            store=store,
            profiles_dir=args.profiles,
        ).recommend(
            args.question,
            query,
            min_candidates=args.min_candidates,
            candidate_limit=args.candidate_limit,
            detail_limit=args.detail_limit,
            expand_profiles=not args.no_expand,
            max_new_profiles=args.max_new_profiles,
            discovery_per_term=args.discovery_per_term,
            enrich_details=args.enrich_details,
            embedder=embedder,
            catalog_path=args.catalog,
            docs_dir=args.docs,
            raw_dir=args.raw,
            index_path=args.index,
            time_analysis_dir=args.time_analysis,
        )
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "analyze-update":
        client = SteamAPIClient()
        embedder = OpenAIEmbedder(args.embedding_model)
        run = run_time_analysis_and_index(
            client=client,
            embedder=embedder,
            game=SteamGame(args.appid, args.name),
            catalog_path=DEFAULT_CATALOG,
            docs_dir=args.docs,
            raw_dir=args.raw,
            profiles_dir=args.profiles,
            index_path=args.index,
            output_dir=args.output_dir,
            before_days=args.before_days,
            after_days=args.after_days,
            max_reviews=args.max_reviews,
            max_pages=args.max_pages,
            news_count=args.news_count,
            patch_date=args.patch_date,
            focus_features=args.focus,
            language=args.language,
        )
        analysis = run.analysis
        print(
            json.dumps(
                {
                    "analysis": str(run.json_path),
                    "markdown": str(run.markdown_path),
                    "vectorstore": str(run.index_path),
                    "appid": args.appid,
                    "name": analysis.game_name,
                    "patch_event": analysis.patch_event.date,
                    "direction": analysis.direction,
                    "confidence": analysis.confidence_label,
                    "before_sample_size": analysis.before.sample_size,
                    "after_sample_size": analysis.after.sample_size,
                    "positive_ratio_delta_pp": analysis.positive_ratio_delta_pp,
                    "indexed": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-stage4":
        cases = load_golden_set(args.golden_set)
        if args.limit > 0:
            cases = cases[: args.limit]
        embedder = OpenAIEmbedder(args.embedding_model)
        needs_generator = not args.retrieval_only or any(
            strategy in {"agentic", "agentic_hyde"} for strategy in args.strategies
        )
        generator = OpenAIAnswerGenerator(args.answer_model) if needs_generator else None
        reranker = (
            CrossEncoderReranker(args.reranker_model)
            if any(strategy in {"reranker", "agentic", "agentic_hyde"} for strategy in args.strategies)
            else None
        )
        records = Stage4BenchmarkRunner(
            VectorIndex.load(args.index),
            embedder,
            generator=generator,
            reranker=reranker,
            top_k=args.top_k,
            rerank_candidates=args.rerank_candidates,
        ).run(cases, args.strategies, generate_answers=not args.retrieval_only)
        summary = save_benchmark(
            records,
            details_path=args.details_output,
            summary_path=args.summary_output,
        )
        print(
            json.dumps(
                {
                    "golden_set": str(args.golden_set),
                    "case_count": len(cases),
                    "strategies": args.strategies,
                    "record_count": len(records),
                    "retrieval_only": args.retrieval_only,
                    "details_output": str(args.details_output),
                    "summary_output": str(args.summary_output),
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-conversations":
        # Imported only for an explicit live evaluation. Validation and unit tests stay offline.
        from steam_rag.application.service_runtime import ServicePaths, SteamServiceRuntime

        started_at = datetime.now(timezone.utc)
        run_id = args.run_id.strip() or started_at.strftime("conversation-%Y%m%dT%H%M%SZ")
        cases = load_conversation_golden_set(args.golden_set)
        if args.limit > 0:
            cases = cases[: args.limit]
        runtime = SteamServiceRuntime(
            paths=ServicePaths(
                docs_dir=args.docs,
                index_path=args.index,
                raw_dir=args.raw,
                catalog_path=args.catalog,
                profiles_dir=args.profiles,
                service_db=args.service_db,
                time_analysis_dir=args.time_analysis,
            ),
            embedding_model=args.embedding_model,
            answer_model=args.answer_model,
        )
        records = ServiceConversationBenchmarkRunner(runtime, top_k=args.top_k).run(cases)
        finished_at = datetime.now(timezone.utc)
        run_metadata = {
            "run_id": run_id,
            "run_label": args.run_label.strip(),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "git_commit": _git_revision(),
            "golden_set": str(args.golden_set),
            "golden_set_sha256": _file_sha256(args.golden_set),
            "cache_state": args.cache_state,
            "case_count": len(cases),
            "turn_count": len(records),
            "top_k": args.top_k,
            "embedding_model": args.embedding_model,
            "answer_model": args.answer_model,
            "reranker_enabled": os.getenv("STEAM_RAG_ENABLE_RERANKER", "1"),
            "reranker_model": os.getenv(
                "STEAM_RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL
            ),
            "python_version": sys.version.split()[0],
        }
        summary = save_conversation_benchmark(
            records,
            details_path=args.details_output,
            summary_path=args.summary_output,
            run_metadata=run_metadata,
        )
        manifest = {
            "run": run_metadata,
            "outputs": {
                "details": str(args.details_output),
                "summary": str(args.summary_output),
                "manifest": str(args.manifest_output),
            },
            "latency_ms": summary.get("latency_ms", {}),
            "operations": summary.get("operations", {}),
        }
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "golden_set": str(args.golden_set),
                    "case_count": len(cases),
                    "turn_count": len(records),
                    "details_output": str(args.details_output),
                    "summary_output": str(args.summary_output),
                    "manifest_output": str(args.manifest_output),
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "collect":
        embedder = OpenAIEmbedder(args.embedding_model)
        update = OnDemandCorpusManager(
            client=SteamAPIClient(),
            catalog_path=DEFAULT_CATALOG,
            docs_dir=args.docs,
            raw_dir=args.raw,
            profiles_dir=args.profiles,
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
                    "profile": str(args.profiles / f"{update.game.resolved_key()}.json"),
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
