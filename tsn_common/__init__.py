"""
TSN Common - Shared models, configuration, and utilities.
"""

__version__ = "2.0.0"

from tsn_common.config import get_settings
from tsn_common.logging import setup_logging

__all__ = ["get_settings", "setup_logging"]
