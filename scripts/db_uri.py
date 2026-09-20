"""Build database URLs safely even when passwords contain URI characters."""
import os
import sys
from urllib.parse import quote


def database_uri(database):
    password = quote(os.environ[f'{database.upper()}_DB_PASSWORD'], safe='')
    return f'postgresql+psycopg2://{database}:{password}@postgres:5432/{database}'


if __name__ == '__main__':
    sys.stdout.write(database_uri(sys.argv[1]))
