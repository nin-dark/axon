import httpx
import pandas as pd
import duckdb
from connectors.base_connector import BaseConnector

class RESTConnector(BaseConnector):

    def __init__(self, endpoint_url: str, auth_token: str = None):
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token

    async def execute(self, query: str) -> list:
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        async with httpx.AsyncClient() as client:
            response = await client.get(self.endpoint_url, headers=headers)
            response.raise_for_status()
            data = response.json()

        df = pd.DataFrame(data)
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(query).fetchall()
        con.close()
        return result