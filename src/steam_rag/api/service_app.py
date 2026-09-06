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
    # §4.4 요청이 어느 공간에서 왔는지가 컨텍스트 선택을 결정한다.
    workspace: str = Field(default="discovery", pattern="^(discovery|play)$")
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    session_id: str = Field(default="", max_length=64)
    game_id: int | None = Field(default=None, gt=0)
    thread_id: str = Field(default="", max_length=64)
    playthrough: int = Field(default=1, ge=1, le=50)


class LibraryRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    appid: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=200)
    header_image: str = Field(default="", max_length=500)
    note: str = Field(default="", max_length=500)


class PreferenceRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    kind: str = Field(pattern="^(like|dislike|played|avoid)$")
    value: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=200)
    evidence: str = Field(default="", max_length=400)
    scope: str = Field(default="persistent", pattern="^(persistent|session)$")
    session_id: str = Field(default="", max_length=64)


class ThreadRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    appid: int = Field(gt=0)
    topic: str = Field(default="general", max_length=40)
    title: str = Field(default="", max_length=120)
    playthrough: int = Field(default=1, ge=1, le=50)


class GameStateRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    appid: int = Field(gt=0)
    playthrough: int = Field(default=1, ge=1, le=50)
    progress: str | None = Field(default=None, max_length=300)
    character_build: str | None = Field(default=None, max_length=200)
    equipment: list[str] | None = Field(default=None, max_length=20)
    difficulties: list[str] | None = Field(default=None, max_length=20)
    spoiler_level: str | None = Field(default=None, pattern="^(no_spoiler|progress|all)$")
    platform: str | None = Field(default=None, max_length=40)
    game_version: str | None = Field(default=None, max_length=60)


class PlaySpaceRequest(BaseModel):
    user_id: str = Field(default="local", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    appid: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=200)
    header_image: str = Field(default="", max_length=500)
    platform: str = Field(default="steam", max_length=40)


class CompareRequest(BaseModel):
    appids: list[int] = Field(min_length=2, max_length=3)


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
            workspace=request.workspace,
            user_id=request.user_id,
            session_id=request.session_id,
            game_id=request.game_id,
            thread_id=request.thread_id,
            playthrough=request.playthrough,
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

    # ------------------------------------------------------------------
    # 내 게임 (§4.5)
    # ------------------------------------------------------------------
    @app.get("/api/library")
    async def library(user_id: str = "local") -> dict[str, Any]:
        return {"games": await asyncio.to_thread(runtime.list_library, user_id)}

    @app.post("/api/library")
    async def save_library_game(request: LibraryRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            runtime.add_library_game,
            request.user_id,
            appid=request.appid,
            name=request.name,
            header_image=request.header_image,
            note=request.note,
        )

    @app.delete("/api/library/{appid}")
    async def delete_library_game(appid: int, user_id: str = "local") -> dict[str, Any]:
        removed = await asyncio.to_thread(runtime.remove_library_game, user_id, appid)
        return {"removed": removed}

    # ------------------------------------------------------------------
    # 내 취향 (§4.5, §11)
    # ------------------------------------------------------------------
    @app.get("/api/preferences")
    async def preferences(user_id: str = "local") -> dict[str, Any]:
        return {"preferences": await asyncio.to_thread(runtime.list_preferences, user_id)}

    @app.post("/api/preferences")
    async def save_preference(request: PreferenceRequest) -> dict[str, Any]:
        saved = await asyncio.to_thread(
            runtime.set_preference,
            request.user_id,
            kind=request.kind,
            value=request.value,
            label=request.label,
            evidence=request.evidence,
            scope=request.scope,
            session_id=request.session_id,
        )
        if saved is None:
            raise HTTPException(status_code=400, detail="저장할 수 없는 취향 값입니다.")
        return saved

    @app.delete("/api/preferences/{preference_id}")
    async def delete_preference(preference_id: int, user_id: str = "local") -> dict[str, Any]:
        removed = await asyncio.to_thread(runtime.delete_preference, user_id, preference_id)
        return {"removed": removed}

    # ------------------------------------------------------------------
    # 게임별 플레이 공간 (§4.4)
    # ------------------------------------------------------------------
    @app.post("/api/play-space")
    async def open_play_space(request: PlaySpaceRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            runtime.open_play_space,
            request.user_id,
            appid=request.appid,
            name=request.name,
            header_image=request.header_image,
            platform=request.platform,
        )

    @app.get("/api/games/{appid}/threads")
    async def play_threads(appid: int, user_id: str = "local") -> dict[str, Any]:
        return {"threads": await asyncio.to_thread(runtime.list_play_threads, user_id, appid)}

    @app.post("/api/games/threads")
    async def create_play_thread(request: ThreadRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            runtime.open_play_thread,
            request.user_id,
            appid=request.appid,
            topic=request.topic,
            title=request.title,
            playthrough=request.playthrough,
        )

    @app.get("/api/games/threads/{thread_id}/messages")
    async def play_thread_messages(thread_id: str, user_id: str = "local") -> dict[str, Any]:
        return {
            "messages": await asyncio.to_thread(runtime.play_thread_messages, user_id, thread_id)
        }

    @app.get("/api/games/{appid}/state")
    async def game_state(appid: int, user_id: str = "local", playthrough: int = 1) -> dict[str, Any]:
        return await asyncio.to_thread(
            runtime.game_state, user_id, appid, playthrough=playthrough
        )

    @app.put("/api/games/{appid}/state")
    async def save_game_state(appid: int, request: GameStateRequest) -> dict[str, Any]:
        if request.appid != appid:
            raise HTTPException(status_code=400, detail="경로와 본문의 appid가 다릅니다.")
        changes = {
            key: value
            for key, value in request.model_dump(exclude={"user_id", "appid"}).items()
            if value is not None
        }
        return await asyncio.to_thread(runtime.update_game_state, request.user_id, appid, **changes)

    @app.post("/api/games/{appid}/playthrough")
    async def new_playthrough(appid: int, user_id: str = "local") -> dict[str, Any]:
        return await asyncio.to_thread(runtime.start_new_playthrough, user_id, appid)

    # ------------------------------------------------------------------
    # 비교 (§4.3, §4.5)
    # ------------------------------------------------------------------
    @app.post("/api/compare")
    async def compare(request: CompareRequest) -> dict[str, Any]:
        return await asyncio.to_thread(runtime.compare, request.appids)

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
