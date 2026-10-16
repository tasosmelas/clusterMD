#!/usr/bin/env bash
set -euo pipefail

echo "=== Initializing Pipeline Environment ==="

# Check if scripts folder exists
if [ ! -d "./scripts" ]; then
    echo "[-] Error: 'scripts/' directory not found."
    echo "[-] Please run this script from the root of the repository."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "[-] Error: 'requirements.txt' not found."
    exit 1
fi

# 1. Get the absolute path to the scripts directory
SCRIPTS_DIR=$(realpath ./scripts)

# 2. Make every file inside the scripts folder executable
echo "[*] Making scripts executable..."
chmod +x "$SCRIPTS_DIR"/*

# 3. Install dependencies cleanly bound to the active Python 3 environment
echo "[*] Ensuring all Python dependencies are installed..."
python3 -m pip install -r requirements.txt

echo "-------------------------------------------------------"
echo "[+] SETUP COMPLETE!"
echo "[+] To run the tools from anywhere, run this command once,"
echo "[+] or add it to your ~/.bashrc file:"
echo ""
echo "export PATH=\"$SCRIPTS_DIR:\$PATH\""
echo "-------------------------------------------------------"
echo "[+] or add it to your ~/.bashrc file:"
echo ""
echo "export PATH=\"$SCRIPTS_DIR:\$PATH\""
echo "-------------------------------------------------------"
