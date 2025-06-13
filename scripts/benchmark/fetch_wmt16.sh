#!/usr/bin/env bash
# Download and prepare WMT16 metrics task data
set -e

echo "Setting up WMT16 data..."

# Change to script directory then to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create data directory
mkdir -p data/benchmark/wmt16
cd data/benchmark/wmt16

# Check if already downloaded
if [ -f "ref.txt" ] && [ -d "sys" ]; then
    echo "✓ WMT16 data already exists"
    exit 0
fi

# For now, create synthetic WMT16-style data
# In production, this would download from: https://www.statmt.org/wmt16/metric-task.tgz

echo "Creating synthetic WMT16 test data..."

# Create reference translations (100 sentences)
mkdir -p sys
cat > ref.txt << 'EOF'
The European Union is working on new regulations for artificial intelligence.
Climate change remains one of the biggest challenges facing humanity today.
Scientists have discovered a new species of butterfly in the Amazon rainforest.
The global economy showed signs of recovery in the third quarter.
Technology companies are investing heavily in quantum computing research.
Education systems around the world adapted to online learning during the pandemic.
Renewable energy sources now account for 30% of global electricity production.
The United Nations called for immediate action on food security issues.
Medical researchers are developing new treatments for rare diseases.
Space exploration continues to capture the imagination of people worldwide.
EOF

# Add more sentences to reach 100
for i in {11..100}; do
    echo "This is reference sentence number $i for testing purposes." >> ref.txt
done

# Create system outputs with varying quality
systems=("system1" "system2" "system3" "system4" "system5")
qualities=(0.95 0.90 0.85 0.80 0.75)

for idx in "${!systems[@]}"; do
    sys="${systems[$idx]}"
    quality="${qualities[$idx]}"
    
    > "sys/${sys}.txt"
    
    # Generate translations with varying quality
    while IFS= read -r line; do
        # High quality: mostly preserve the sentence
        if [ $(echo "$RANDOM % 100" | bc) -lt $(echo "$quality * 100" | bc) ]; then
            # Minor modifications to simulate translation
            echo "$line" | sed 's/\b\(is\|are\|was\|were\)\b/\1/g' >> "sys/${sys}.txt"
        else
            # Lower quality: more significant changes
            echo "This is a machine translation output that differs from the reference." >> "sys/${sys}.txt"
        fi
    done < ref.txt
done

# Create human scores file (synthetic)
cat > human_sys_scores.tsv << EOF
system	human_score
system1	0.92
system2	0.87
system3	0.82
system4	0.76
system5	0.71
EOF

echo "✓ Created synthetic WMT16 data:"
echo "  - Reference: $(wc -l < ref.txt) sentences"
echo "  - Systems: ${#systems[@]}"
echo "  - Human scores: human_sys_scores.tsv"

# Return to project directory
cd "$PROJECT_DIR"