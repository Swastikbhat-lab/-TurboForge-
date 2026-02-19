#!/bin/bash

# ─────────────────────────────────────────────
# TurboForge - One-Click Setup Script
# Run: bash setup.sh
# ─────────────────────────────────────────────

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   TurboForge - Automated Setup Script   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Step 1: Check Python ───────────────────────
echo "▶ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found: Python $python_version"

# ── Step 2: Create folder structure ───────────
echo ""
echo "▶ Creating project structure..."
mkdir -p data models utils

touch data/__init__.py
touch models/__init__.py
touch utils/__init__.py

echo "  ✅ Folders and __init__.py files created"

# ── Step 3: Install dependencies ──────────────
echo ""
echo "▶ Installing dependencies (this may take a few minutes)..."

# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    echo "  PyTorch already installed, skipping..."
else
    echo "  Installing PyTorch..."
    pip install torch --quiet
fi

pip install numpy pandas scikit-learn matplotlib seaborn tqdm anthropic --quiet
echo "  ✅ All dependencies installed"

# ── Step 4: Set API key ────────────────────────
echo ""
echo "▶ Setting Anthropic API key..."

export ANTHROPIC_API_KEY="YOUR_NEW_API_KEY_HERE"

# Also write to shell profile so it persists
SHELL_PROFILE=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_PROFILE="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_PROFILE="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_PROFILE="$HOME/.bash_profile"
fi

if [ -n "$SHELL_PROFILE" ]; then
    # Remove old key if exists
    grep -v "ANTHROPIC_API_KEY" "$SHELL_PROFILE" > /tmp/profile_tmp && mv /tmp/profile_tmp "$SHELL_PROFILE"
    echo "export ANTHROPIC_API_KEY=\"YOUR_NEW_API_KEY_HERE\"" >> "$SHELL_PROFILE"
    echo "  ✅ API key saved to $SHELL_PROFILE (persists across terminal sessions)"
else
    echo "  ✅ API key set for this session"
fi

# ── Step 5: Verify setup ───────────────────────
echo ""
echo "▶ Verifying setup..."
python3 -c "
import torch, numpy, pandas, sklearn, anthropic
print('  ✅ torch:', torch.__version__)
print('  ✅ numpy:', numpy.__version__)
print('  ✅ pandas:', pandas.__version__)
print('  ✅ anthropic: OK')
"

# ── Step 6: Init git repo ──────────────────────
echo ""
echo "▶ Initializing Git repository..."

if [ ! -d ".git" ]; then
    git init
    cat > .gitignore << 'EOF'
__pycache__/
*.pt
*.pth
*.csv
*.pyc
.env
*.egg-info/
dist/
build/
.DS_Store
EOF
    git add .
    git commit -m "TurboForge: Generative AI digital twin for wind farm failure prediction"
    echo "  ✅ Git repo initialized and initial commit made"
else
    echo "  Git repo already exists, skipping init"
fi

# ── Step 7: Quick smoke test ───────────────────
echo ""
echo "▶ Running quick smoke test (5 epochs, no GAN)..."
python3 main.py --mode train --epochs 5 --skip_gan
echo "  ✅ Smoke test passed!"

# ── Done ───────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           ✅ Setup Complete!             ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Quick test:  python3 main.py --mode full --epochs 5 --skip_gan"
echo "║  Full train:  python3 main.py --mode full --epochs 50"
echo "║  Real data:   python3 main.py --mode full --data_csv your_file.csv"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "⚠️  IMPORTANT: Run 'source ~/.zshrc' (or ~/.bashrc) to load your API key"
echo "    in any existing terminal windows."
echo ""
