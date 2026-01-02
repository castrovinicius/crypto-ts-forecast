"""Main entry point for running the FastAPI server."""

import argparse
import logging
from pathlib import Path

import uvicorn

from crypto_ts_forecast.api import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(
        description="Bitcoin Price Forecast API Server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to the Kedro project (default: current directory)",
    )

    args = parser.parse_args()

    project_path = Path(args.project_path) if args.project_path else Path.cwd()

    logger.info(f"Starting API server at http://{args.host}:{args.port}")
    logger.info(f"Project path: {project_path}")
    logger.info("API documentation available at /docs")

    # Create app with project path
    app = create_app(project_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
