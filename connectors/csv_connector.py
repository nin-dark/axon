import pandas as pd
import duckdb
from connectors.base_connector import BaseConnector

class CSVConnector(BaseConnector):

    def __init__(self, file_path: str):
        self.file_path = file_path

    async def execute(self, query: str) -> list:
        df = pd.read_csv(self.file_path)
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(query).fetchall()
        con.close()
        return result