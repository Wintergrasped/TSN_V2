# Database Migration Guide (Debian 12 + Docker)

This guide walks a Debian 12 operator with limited Linux experience through the
steps required to update the TSN database schema after pulling new code or images.
It assumes the stack runs via Docker Compose on the same host that stores the
database (MySQL/MariaDB) and that you have shell access with sudo.

---

## 1. Before you start

1. **Know your database**
   - MySQL/MariaDB default port 3306.
   - Credentials live in `.env` or environment variables referenced by
     `TSN_DB_*` settings. You will need them for backups.

2. **Update the repo and images**
   ```bash
   cd ~/TSN_V2
   git pull origin main
   docker compose pull
   docker compose build --no-cache tsn_server
   ```

3. **Stop running containers** (keeps writes from happening mid-migration)
   ```bash
   docker compose down
   ```

4. **Install helper packages (only once)**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip mariadb-client
   ```

---

## 2. Take a backup (strongly recommended)

> Skip only if this is a disposable/test database.

### MySQL/MariaDB example
```bash
mysqldump -h <db-host> -u <db-user> -p<db-password> <db-name> \
  > ~/tsn_backup_$(date +%Y%m%d%H%M).sql
```

Keep the backup file somewhere safe before continuing.

---

## 3. Option A – Apply schema migrations (recommended)

Use this when you already have live data you want to keep.

1. **Bring the TSN services back up but keep external traffic blocked**
   ```bash
   docker compose up -d tsn_server
   ```

2. **Open a shell inside the server container**
   ```bash
   docker compose exec tsn_server bash
   ```

3. **Run the Alembic migration**
   ```bash
   cd /app
   poetry run alembic upgrade head
   # or, if poetry isn't installed, fall back to python -m alembic upgrade head
   ```
   - This updates existing tables (`net_sessions`, `callsign_profiles`, etc.) and
     creates new ones (`club_profiles`, `club_memberships`, `trend_snapshots`).

4. **Exit the container**
   ```bash
   exit
   ```

5. **Restart the full stack**
   ```bash
   docker compose down
   docker compose up -d
   ```

### Verifying
```bash
mysql -h <db-host> -u <db-user> -p<db-password> -e \
   "SHOW TABLES LIKE 'club_profiles';" <db-name>
```
Tables should appear with the current timestamp.

---

## 4. Option B – Recreate the database (only if data can be dropped)

This is simpler but wipes everything. Use for new installs or when you do not
need prior recordings.

1. **Drop and recreate the database** using your DB admin tool or CLI.
2. **Run the init script inside the server container**
   ```bash
   docker compose exec tsn_server bash -c "cd /app && poetry run python -m tsn_common.db_init"
   ```
   - Creates all tables and seeds phonetic corrections.
3. **Restart services**
   ```bash
   docker compose down
   docker compose up -d
   ```

---

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `alembic: command not found` | Use `poetry run alembic` or install with `pip install alembic` inside the container. |
| `permission denied for relation ...` | Ensure the DB user has `ALTER`/`CREATE` privileges. Run migration using a privileged user. |
| `relation "club_profiles" already exists` | Migration ran previously. Safe to continue; verify with `alembic current`. |
| `Can't connect to local MySQL server` | Confirm `docker compose ps` shows the DB container healthy, or supply the correct host/port when DB runs elsewhere. |

---

## 6. After migration

1. Tail logs to ensure the analyzer sees the new schema:
   ```bash
   docker compose logs -f tsn_server
   ```
2. Look for `analysis_batch_completed` log entries without SQL errors.
3. If anything fails, restore from the backup taken earlier and retry.

You now have the database ready for the upgraded vLLM analysis pipeline.
