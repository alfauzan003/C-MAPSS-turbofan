-- Create the integration test database alongside the main one.
-- Idempotent: do nothing if it already exists.
SELECT 'CREATE DATABASE pdm_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pdm_test')\gexec
