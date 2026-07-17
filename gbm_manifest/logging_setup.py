"""Configure logging once for the pipeline. Library code never calls this."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=3
            )
        )

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers,
                        force=True)
    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "nibabel"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
