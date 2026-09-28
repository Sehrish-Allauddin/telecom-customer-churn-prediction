"""Central logging configuration for the churn project."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent application logging without recording customer inputs.

    Args:
        level: Minimum severity to emit.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
