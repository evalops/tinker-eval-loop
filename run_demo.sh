#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Tinker Evaluation-Driven Fine-Tuning Demo               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ -z "$TINKER_API_KEY" ]; then
    echo "❌ Error: TINKER_API_KEY environment variable not set"
    echo ""
    echo "Please set your Tinker API key:"
    echo "  export TINKER_API_KEY=sk-your-key-here"
    echo ""
    exit 1
fi

echo "✓ Tinker API key found"
echo ""

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "📦 Installing dependencies..."
source .venv/bin/activate
pip install -q -r requirements.txt

echo "✓ Dependencies installed"
echo ""

echo "🚀 Starting evaluation-driven training loop..."
echo "   - Base model: meta-llama/Llama-3.1-8B-Instruct"
echo "   - Training data: 20 examples (demo_data.jsonl)"
echo "   - Max rounds: 3"
echo "   - Eval threshold: 0.75"
echo "   - Initial LR: 0.0003 (decays by 0.6x per round)"
echo ""

python trainer_with_eval.py --config demo_config.json

echo ""
echo "✅ Demo complete!"
echo ""
echo "What happened:"
echo "  1. Loaded 20 training examples"
echo "  2. Fine-tuned model with LoRA on Tinker infrastructure"
echo "  3. Evaluated model on QA tasks"
echo "  4. If score < 0.75: adjusted LR and started Round 2"
echo "  5. Repeated until threshold met or max rounds reached"
echo ""
echo "Next steps:"
echo "  - Review checkpoints saved in Tinker"
echo "  - Enable EvalOps integration to track all runs"
echo "  - Replace simple_eval.py with real Inspect AI tasks"
