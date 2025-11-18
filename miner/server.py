import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fiber.logging_utils import get_logger
from fiber.miner.core import configuration

from miner.endpoints.tuning import factory_router as tuning_factory_router


logger = get_logger(__name__)


def factory_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()

        yield

        logger.info("Shutting down...")

        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    return app


logger = get_logger(__name__)

app = factory_app(debug=True)


tuning_router = tuning_factory_router()

app.include_router(tuning_router)

# if os.getenv("ENV", "prod").lower() == "dev":
#    configure_extra_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7999)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "who": "Ree928",
        "component": "miner",
        "version": "tao-v1"
    }

from fastapi import Request
import logging
import time

logger = logging.getLogger("god")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    status_code = response.status_code

    logger.info(
        f"{client_host} {method} {path} -> {status_code} in {duration_ms:.2f} ms"
    )

    return response

from fastapi import Request
import logging
import time

logger = logging.getLogger("god")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    status_code = response.status_code

    logger.info(
        f"{client_host} {method} {path} -> {status_code} in {duration_ms:.2f} ms"
    )

    return response

from fastapi import Request
import logging
import time

logger = logging.getLogger("god")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    client_host = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    status_code = response.status_code

    logger.info(
        f"{client_host} {method} {path} -> {status_code} in {duration_ms:.2f} ms"
    )

    return response
