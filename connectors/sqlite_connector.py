import aiosqlite
import asyncio
from connectors.base_connector import BaseConnector

class SQLiteConnector(BaseConnector):

    def __init__(self, db_path: str, semaphore: asyncio.Semaphore):
        self.db_path = db_path
        self.semaphore = semaphore

    async def execute(self, query: str) -> list:
        async with self.semaphore:
             async with aiosqlite.connect(self.db_path) as db:
                 async with db.execute(query) as cursor:
                     return await cursor.fetchall()