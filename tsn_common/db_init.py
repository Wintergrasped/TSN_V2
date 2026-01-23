"""
Database initialization and migration setup.
"""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from tsn_common.config import get_settings
from tsn_common.db import get_engine
from tsn_common.logging import get_logger
from tsn_common.models.base import Base

logger = get_logger(__name__)


async def create_all_tables() -> None:
    """Create all tables from SQLAlchemy models."""
    settings = get_settings()
    engine = get_engine(settings.database)
    
    logger.info("creating_database_tables")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("database_tables_created")
    
    await engine.dispose()


async def seed_phonetic_corrections() -> None:
    """Seed phonetic corrections table."""
    from tsn_common.db import get_session
    from tsn_common.models.support import PhoneticCorrection
    
    corrections = [
        # Common misheard letters
        ("alpha", "A"),
        ("bravo", "B"),
        ("charlie", "C"),
        ("delta", "D"),
        ("echo", "E"),
        ("foxtrot", "F"),
        ("golf", "G"),
        ("hotel", "H"),
        ("india", "I"),
        ("juliet", "J"),
        ("kilo", "K"),
        ("lima", "L"),
        ("mike", "M"),
        ("november", "N"),
        ("oscar", "O"),
        ("papa", "P"),
        ("quebec", "Q"),
        ("romeo", "R"),
        ("sierra", "S"),
        ("tango", "T"),
        ("uniform", "U"),
        ("victor", "V"),
        ("whiskey", "W"),
        ("xray", "X"),
        ("x-ray", "X"),
        ("yankee", "Y"),
        ("zulu", "Z"),
        # Common Whisper errors
        ("won", "1"),
        ("too", "2"),
        ("to", "2"),
        ("tree", "3"),
        ("three", "3"),
        ("for", "4"),
        ("four", "4"),
        ("fife", "5"),
        ("five", "5"),
        ("sex", "6"),
        ("six", "6"),
        ("seven", "7"),
        ("ait", "8"),
        ("eight", "8"),
        ("niner", "9"),
        ("nine", "9"),
        ("zero", "0"),
        # Specific callsign patterns
        ("w won", "W1"),
        ("w too", "W2"),
        ("w tree", "W3"),
        ("k won", "K1"),
        ("k too", "K2"),
        ("n won", "N1"),
        ("a a", "AA"),
        ("triple a", "AAA"),
        ("stroke", "/"),
        ("slash", "/"),
        ("mobile", "/M"),
        ("portable", "/P"),
    ]
    
    logger.info("seeding_phonetic_corrections", count=len(corrections))
    
    async with get_session() as session:
        for misheard, correct in corrections:
            correction = PhoneticCorrection(
                misheard_text=misheard.lower(),
                correct_text=correct,
            )
            session.add(correction)
    
    logger.info("phonetic_corrections_seeded")


def init_alembic() -> None:
    """Initialize Alembic configuration."""
    project_root = Path(__file__).parent.parent
    alembic_dir = project_root / "migrations"
    
    if not alembic_dir.exists():
        logger.info("initializing_alembic")
        
        # Create alembic config
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://user:pass@localhost/tsn")
        
        # Initialize alembic
        command.init(alembic_cfg, str(alembic_dir))
        
        logger.info("alembic_initialized")
    else:
        logger.info("alembic_already_initialized")


async def main() -> None:
    """Main entry point."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    # Create tables
    await create_all_tables()
    
    # Seed data
    await seed_phonetic_corrections()
    
    logger.info("database_initialization_complete")


if __name__ == "__main__":
    asyncio.run(main())
