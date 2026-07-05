from abc import ABC, abstractmethod


class BaseConnector(ABC):

    @abstractmethod
    async def execute(self, query: str) -> list:
        """Execute a SQL query and return results as a list of tuples."""
        ...