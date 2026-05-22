# DinoBoard AI Deployment

This directory contains production templates for the same-domain AI API.

1. Install DinoBoard under `/opt/dinoboard-ai/DinoBoard`.
2. Create a Python venv and install runtime dependencies plus the editable package.
3. Install `deploy/dinoboard-ai.service` into `/etc/systemd/system/`.
4. Add `deploy/caddy-dinoboard-ai.snippet` before `file_server` in each Caddy site block.
5. Install the fail2ban files:
   - `deploy/fail2ban/filter.d/dinoboard-ai.conf` -> `/etc/fail2ban/filter.d/dinoboard-ai.conf`
   - `deploy/fail2ban/jail.d/dinoboard-ai.local` -> `/etc/fail2ban/jail.d/dinoboard-ai.local`

Validate with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dinoboard-ai.service
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl restart fail2ban
curl http://127.0.0.1:8001/api/games/available
curl https://zephyrlabs.cloud/api/dinoboard/api/games/available
```

Manual unban:

```bash
sudo fail2ban-client set dinoboard-ai unbanip <ip>
```
