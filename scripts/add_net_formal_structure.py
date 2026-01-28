"""
Manual migration to add formal structure columns to net_sessions table.
Run this if auto-migration didn't add the columns.
"""

import asyncio
from sqlalchemy import text
from tsn_common.db import async_session_maker
from tsn_common.logging import get_logger

logger = get_logger(__name__)


async def add_formal_structure_columns():
    """Add formal_structure, ncs_script, and checkin_sequence columns."""
    
    async with async_session_maker() as session:
        try:
            # Check if columns already exist
            check_sql = """
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'net_sessions' 
                AND COLUMN_NAME IN ('formal_structure', 'ncs_script', 'checkin_sequence')
            """
            result = await session.execute(text(check_sql))
            existing_columns = {row[0] for row in result.fetchall()}
            
            if len(existing_columns) == 3:
                logger.info("formal_structure_columns_exist", 
                           message="All columns already exist, no migration needed")
                return
            
            logger.info("formal_structure_migration_starting", 
                       existing_columns=list(existing_columns))
            
            # Add columns that don't exist
            if 'formal_structure' not in existing_columns:
                logger.info("adding_column", column="formal_structure")
                await session.execute(text(
                    "ALTER TABLE net_sessions ADD COLUMN formal_structure JSON NULL"
                ))
                await session.commit()
                logger.info("column_added", column="formal_structure")
            
            if 'ncs_script' not in existing_columns:
                logger.info("adding_column", column="ncs_script")
                await session.execute(text(
                    "ALTER TABLE net_sessions ADD COLUMN ncs_script JSON NULL"
                ))
                await session.commit()
                logger.info("column_added", column="ncs_script")
            
            if 'checkin_sequence' not in existing_columns:
                logger.info("adding_column", column="checkin_sequence")
                await session.execute(text(
                    "ALTER TABLE net_sessions ADD COLUMN checkin_sequence JSON NULL"
                ))
                await session.commit()
                logger.info("column_added", column="checkin_sequence")
            
            logger.info("formal_structure_migration_complete")
            
        except Exception as exc:
            logger.error("formal_structure_migration_failed", error=str(exc))
            raise


if __name__ == "__main__":
    print("🔧 Adding formal structure columns to net_sessions table...")
    asyncio.run(add_formal_structure_columns())
    print("✅ Migration complete!")
