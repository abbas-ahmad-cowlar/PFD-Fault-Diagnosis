#!/bin/bash

# Generate Advanced Figures (Fig17 and Fig18) using Python
# These create beautiful, professional, publication-quality visualizations

echo "========================================================================="
echo "  GENERATING ADVANCED FIGURES (Fig17 & Fig18)"
echo "========================================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    exit 1
fi

# Check required packages
echo "Checking Python dependencies..."
python3 -c "import numpy, matplotlib, seaborn, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing required packages..."
    pip3 install numpy matplotlib seaborn scipy --quiet
fi

echo "✓ Dependencies OK"
echo ""

# Generate Fig17
echo "Generating Fig17 (Random Forest Analysis)..."
python3 generate_fig17_rf_analysis.py
if [ $? -eq 0 ]; then
    echo "✓ Fig17 complete"
else
    echo "❌ Fig17 failed"
fi

echo ""

# Generate Fig18
echo "Generating Fig18 (Neural Network Analysis)..."
python3 generate_fig18_nn_analysis.py
if [ $? -eq 0 ]; then
    echo "✓ Fig18 complete"
else
    echo "❌ Fig18 failed"
fi

echo ""
echo "========================================================================="
echo "✅ DONE! Check your results directory for the new figures."
echo "========================================================================="
