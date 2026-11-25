#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALLOW_NON_ARM=false
SKIP_APT=false

for arg in "$@"; do
    case "$arg" in
        --allow-non-arm) ALLOW_NON_ARM=true ;;
        --skip-apt) SKIP_APT=true ;;
        *)
            echo "Argumen tidak dikenal: $arg"
            echo "Penggunaan: $0 [--allow-non-arm] [--skip-apt]"
            exit 1
            ;;
    esac
done

ARCH="$(uname -m)"
echo "=== NutriScan IoT Setup (dir: $PROJECT_DIR, arch: $ARCH) ==="

if [[ "$ARCH" != "aarch64" && "$ALLOW_NON_ARM" != "true" ]]; then
    echo "Peringatan: Sistem target adalah Raspberry Pi OS 64-bit (aarch64)."
    echo "Jika ingin menguji instalasi di x86_64/laptop, tambahkan --allow-non-arm."
    exit 1
fi

if [[ "$SKIP_APT" != "true" ]] && command -v apt-get >/dev/null 2>&1; then
    echo "[1/4] Memasang paket sistem Raspberry Pi OS (picamera2, lgpio, sqlite3)..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-venv \
        python3-pip \
        python3-picamera2 \
        python3-lgpio \
        libgl1 \
        libglib2.0-0 \
        sqlite3
else
    echo "[1/4] Melewati apt-get (--skip-apt atau apt-get tidak tersedia)."
fi

echo "[2/4] Membuat virtual environment (.venv --system-site-packages)..."
python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"

echo "[3/4] Memasang dependensi Python..."
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "[4/4] Menyiapkan direktori kerja & berkas konfigurasi awal..."
mkdir -p "$PROJECT_DIR/data/captures" "$PROJECT_DIR/logs"

if [[ ! -f "$PROJECT_DIR/data/config.json" ]]; then
    cp "$PROJECT_DIR/data/config.example.json" "$PROJECT_DIR/data/config.json"
    echo "Dibuat: data/config.json (silakan sesuaikan mode, pin, & endpoint)."
fi

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
    echo "Dibuat: .env (chmod 600 — isi kredensial produksi Anda)."
fi

echo ""
echo "Instalasi selesai!"
echo "Langkah selanjutnya:"
echo "  1. source .venv/bin/activate"
echo "  2. python scripts/check_env.py"
echo "  3. python scripts/calibrate_hx711.py"
echo "  4. python main.py"
