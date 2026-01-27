# TSN Web Interface Deployment Guide (Debian 12)

This document walks an operator with limited Linux experience through deploying the
new FastAPI-based web portal (replacement for the legacy `kk7nqn.net` site) onto a
remote Debian 12 host. The guide assumes you already have SSH access to the host
and valid database credentials for the existing MySQL/PostgreSQL instance that
TSN uses for ingestion.

> **At a glance**
> 1. Install OS prerequisites (Python, git, build tooling).
> 2. Clone the repository and create a Python virtual environment.
> 3. Copy `.env.example` to `.env`, set TSN + TSN_WEB variables, and verify DB connectivity.
> 4. Launch the portal with `uvicorn` for a quick test.
> 5. (Recommended) Create a `systemd` service and reverse proxy so the site auto-starts.

---

## 1. Prepare the server

1. SSH into the host (replace `radio.example.com` with your server):
   ```bash
   ssh user@radio.example.com
   ```
2. Update package indexes and install required packages:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip git build-essential \
       libffi-dev libssl-dev nginx
   ```
3. (Optional) Create a dedicated user so the web app does not run as `root`:
   ```bash
   sudo adduser --system --group tsn
   sudo mkdir -p /opt/tsn
   sudo chown tsn:tsn /opt/tsn
   ```

## 2. Fetch the project and set up Python

1. Move into your working directory and clone the repository:
   ```bash
   cd /opt/tsn    # or another path you control
   sudo -u tsn git clone https://github.com/Wintergrasped/TSN_V2.git
   cd TSN_V2
   ```
2. Create and activate a virtual environment (stays inside the repo):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```
   This installs FastAPI, uvicorn, passlib, SQLAlchemy drivers, etc.

## 3. Configure environment variables

1. Copy the example environment file and edit it:
   ```bash
   cp .env.example .env
   nano .env
   ```
2. Update the database block to point at the **existing** TSN database (MySQL example shown):
   ```dotenv
   TSN_DB_ENGINE=mysql
   TSN_DB_HOST=51.81.202.9
   TSN_DB_PORT=3306
   TSN_DB_NAME=repeater
   TSN_DB_USER=server
   TSN_DB_PASSWORD=wowhead1
   ```
3. Configure the web portal section:
   ```dotenv
   TSN_WEB_SESSION_SECRET="change-this"   # any long random string
   TSN_WEB_BRAND_NAME="KK7NQN"
   TSN_WEB_SUPPORT_EMAIL="support@yourdomain"
   TSN_WEB_ALLOW_REGISTRATION=true        # set false if you plan to invite users manually
   ```
4. (Optional) Adjust branding, logging, or transcription settings as needed. Save the file.

## 4. Initialize database tables (only once)

The portal introduces a new `portal_users` table for login/registration. The FastAPI
app automatically creates the table on startup, but you can also run this manually:
```bash
source .venv/bin/activate
python -c "from tsn_common.db import get_engine; from tsn_common.models.base import Base; import asyncio; async def main(): engine = get_engine(); async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all); asyncio.run(main())"
```
Most operators simply start the app, which performs the same operation during the
`startup` event.

## 5. Test the portal with uvicorn

1. Start the server in development mode:
   ```bash
   source .venv/bin/activate
   uvicorn web.main:app --host 0.0.0.0 --port 8080 --reload
   ```
2. Visit `http://<server-ip>:8080/` in a browser. You should see the new dashboard
   with queue depth, callsigns, nets, and other widgets.
3. Press `Ctrl+C` to stop the server once you confirm it works.

## 5b. Run the portal with Docker Compose (optional)

If you prefer containers, the repository now ships a dedicated `tsn_web` target and
Compose service so the portal no longer shares the heavyweight TSN server image.

1. Copy `.env.example` to `.env` (if you have not already) and fill in the database
   and `TSN_WEB_*` variables. The compose file reuses `TSN_DB_*` for every service.
2. Build and start only the web portal:
   ```bash
   docker compose up -d tsn_web
   ```
   This command uses the lightweight FastAPI image and exposes the app on
   `TSN_WEB_PORT` (defaults to 8081 on the host).
3. (Optional) Run the web portal alongside `tsn_server` so the analyzer and UI share
   the same stack:
   ```bash
   docker compose up -d tsn_server tsn_web
   ```
4. Tail logs whenever needed:
   ```bash
   docker compose logs -f tsn_web
   ```

---

## 6. Create a `systemd` service (recommended)

1. Create `/etc/systemd/system/tsn-web.service` with the following content:
   ```ini
   [Unit]
   Description=TSN Web Portal
   After=network.target

   [Service]
   User=tsn
   Group=tsn
   WorkingDirectory=/opt/tsn/TSN_V2
   Environment="PATH=/opt/tsn/TSN_V2/.venv/bin"
   EnvironmentFile=/opt/tsn/TSN_V2/.env
   ExecStart=/opt/tsn/TSN_V2/.venv/bin/uvicorn web.main:app --host 0.0.0.0 --port 8080
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
2. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now tsn-web.service
   sudo systemctl status tsn-web.service
   ```
   You should see `Active: active (running)` if everything succeeds.

## 7. Add an Nginx reverse proxy (optional but best practice)

1. Create `/etc/nginx/sites-available/tsn-web`:
   ```nginx
   server {
       listen 80;
       server_name kk7nqn.net;

       location / {
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /static/ {
           alias /opt/tsn/TSN_V2/web/static/;
       }
   }
   ```
2. Enable the site and reload Nginx:
   ```bash
   sudo ln -s /etc/nginx/sites-available/tsn-web /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```
3. (Optional) Obtain free HTTPS certificates via [Certbot](https://certbot.eff.org/)
   and update the server block to listen on port 443.

## 8. Creating the first user

1. Ensure the portal is running.
2. Visit `http://kk7nqn.net/register` (or the host/IP) and create an account.
3. The first user can manually upgrade others to admin by running a SQL update:
   ```sql
   UPDATE portal_users SET is_admin = 1 WHERE email = 'you@example.com';
   ```
   (Use `mysql` or `psql` depending on your database engine.)

## 9. Ongoing maintenance

- **Update code**:
  ```bash
  cd /opt/tsn/TSN_V2
  git pull origin main
  source .venv/bin/activate && pip install -e .
  sudo systemctl restart tsn-web
  ```
- **Logs**: check `journalctl -u tsn-web -f` for runtime output.
- **Backups**: keep `.env` and any TLS certificates backed up; no extra data is stored
  on disk besides logs.
- **Resource monitoring**: watch `top` or `htop` to ensure CPU/memory usage stays low.

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `uvicorn` exits immediately with DB errors | Verify `.env` database credentials and ensure the DB host is reachable (firewall rules, `ping`, `telnet host 3306`). |
| Cannot import `passlib` | Re-run `pip install -e .` inside the virtual environment. |
| `systemctl status tsn-web` shows permission errors | Ensure the `User`/`Group` in the service file own the project path and virtual environment, or switch the service to run as your current user. |
| Browser can’t load CSS/JS | Confirm Nginx `location /static/` path matches `/opt/tsn/TSN_V2/web/static/`. |
| Registration should be disabled | Set `TSN_WEB_ALLOW_REGISTRATION=false` and restart the service; only existing accounts will work. |

You should now have the modern TSN web interface online, backed by the same
database as ingestion and analyzer services. Reach out to the TSN maintainers if
you need help extending the dashboard or integrating additional auth providers.
