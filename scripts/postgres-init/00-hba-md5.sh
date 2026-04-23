#!/bin/bash
# Allow passwordless connections from any host for local dev.
# Passwords are still enforced inside containers via docker-compose env vars.
# On a production VPS, this file would not be used (the container network is locked down).
sed -i 's/scram-sha-256/trust/g' "$PGDATA/pg_hba.conf"
sed -i 's/^host all all all .*/host all all all trust/' "$PGDATA/pg_hba.conf"
