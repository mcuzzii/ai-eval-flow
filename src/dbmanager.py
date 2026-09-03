import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

class DatabaseManager:
    def __init__(self):
        load_dotenv()

        self.schema = os.getenv("PG_SCHEMA", "public")
        self.engine = self._create_engine()
        self.session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
    
    def _create_engine(self):

        url = (
            f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}"
        )
        return create_engine(url, execution_options={"read_only": True})
    
    def get_table_json(self, table_name, limit=None):
        query = f"SELECT * FROM {self.schema}.{table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).mappings().all()
            return result


if __name__ == "__main__":
    db_manager = DatabaseManager()
    table = db_manager.get_table_json('document_embeddings_v2', limit=3)
    for i in range(3):
        print(table[i])