#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(whoami)"
SERVICE_TMP="/tmp/nutriscan.service"

echo "Menyiapkan NutriScan systemd autostart..."
echo "Project Directory: $PROJECT_DIR"
echo "User: $USER_NAME"

cat > "$SERVICE_TMP" << EOF
[Unit]
Description=NutriScan MBG Edge AI & Load Cell IoT Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
SupplementaryGroups=gpio video
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONFAULTHANDLER=1
Environment=OPENBLAS_CORETYPE=ARMV8
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=2
Environment=OMP_WAIT_POLICY=PASSIVE
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/main.py --config $PROJECT_DIR/data/config.json
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo cp "$SERVICE_TMP" /etc/systemd/system/nutriscan.service
rm -f "$SERVICE_TMP"
sudo systemctl daemon-reload
sudo systemctl enable nutriscan.service

echo "Autostart NutriScan berhasil dipasang!"
echo "Jalankan sekarang : sudo systemctl start nutriscan"
echo "Lihat log live    : sudo journalctl -u nutriscan -f"
