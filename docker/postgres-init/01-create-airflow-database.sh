#!/usr/bin/env bash
# Airflow keeps its own metadata database.  Sharing one database between the
# scheduler's bookkeeping and the research warehouse would mean a schema
# migration in one could lock tables in the other, and it makes backups
# ambiguous.  Same server, separate databases.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE airflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
    GRANT ALL PRIVILEGES ON DATABASE airflow TO $POSTGRES_USER;
EOSQL
