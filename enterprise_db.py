import logging
from sqlalchemy import create_engine, pool, text

logger = logging.getLogger("axon.enterprise_db")

class EnterpriseDatabase:
    def __init__(self, dsn: str):
        self.dsn = dsn
        
        # Build the high-speed connection pool
        self.engine = create_engine(
            dsn,
            poolclass=pool.QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_timeout=30,
            pool_recycle=1800  # recycle connections after 30 mins
        )
        logger.info(f"Initialized Enterprise DB Pool (QueuePool) for DSN")

    def execute_query(self, sql: str):
        """
        Executes a SQL query using the connection pool.
        Impersonation support is under development and will be added in a future release.
        """
        with self.engine.begin() as connection:
            result = connection.execute(text(sql))
            if result.returns_rows:
                return [dict(r) for r in result.mappings()]
            else:
                return [{"status": "Success", "rows_affected": result.rowcount}]

