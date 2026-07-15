from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from steam_rag.application.service_runtime import ServicePaths, SteamServiceRuntime
from steam_rag.external_apis.openai_client import load_env_file


WEB_DIR = Path(__file__).resolve().parents[1] / "ui" / "web"


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class ContextGame(BaseModel):
    appid: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=200)


class ConversationState(BaseModel):
    """Small client-owned state used to make the stateless API conversation-safe."""

    active_games: list[ContextGame] = Field(default_factory=list, max_length=10)
    last_mode: str = Field(default="", max_length=32)
    last_resolved_question: str = Field(default="", max_length=1600)
    recommendation_query: dict[str, Any] = Field(default_factory=dict)
    similarity_spec: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    top_k: int = Field(default=6, ge=1, le=10)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    context_games: list[ContextGame] = Field(default_factory=list, max_length=10)
    conversation_state: ConversationState = Field(default_factory=ConversationState)
    request_id: str | None = Field(default=None, min_length=8, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")


def create_service_app(runtime: SteamServiceRuntime | None = None) -> FastAPI:
    runtime = runtime or SteamServiceRuntime()
    app = FastAPI(title="SteamLens AI", version="0.1.0")
    chat_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}
    chat_results: dict[str, dict[str, Any]] = {}

    async def execute_chat(request: ChatRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            runtime.ask,
            request.question,
            top_k=request.top_k,
            history=[message.model_dump() for message in request.history],
            context_games=[game.model_dump() for game in request.context_games],
            conversation_state=request.conversation_state.model_dump(),
        )

    async def execute_cached_chat(request: ChatRequest) -> dict[str, Any]:
        request_id = str(request.request_id)
        try:
            result = await execute_chat(request)
            chat_results[request_id] = result
            while len(chat_results) > 50:
                chat_results.pop(next(iter(chat_results)))
            return result
        finally:
            chat_tasks.pop(request_id, None)

    @app.middleware("http")
    async def disable_prototype_asset_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return await asyncio.to_thread(runtime.health)

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            if not request.request_id:
                return await execute_chat(request)
            if request.request_id in chat_results:
                return chat_results[request.request_id]
            task = chat_tasks.get(request.request_id)
            if task is None:
                task = asyncio.create_task(execute_cached_chat(request))
                chat_tasks[request.request_id] = task
            return await asyncio.shield(task)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
    return app


app = create_service_app()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the SteamLens consumer website")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_env_file(args.env_file)
    if args.inbrowser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    import uvicorn

    uvicorn.run(
        "steam_rag.api.service_app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
