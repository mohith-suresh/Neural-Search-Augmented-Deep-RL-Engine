#!/usr/bin/env python3
'''
FINAL CORRECTED Chart Generation Script for EE542 Presentation
USC Viterbi School of Engineering - Fall 2025
Team: Krish Modi, Mohith Suresh, Adithya Srivastava

Creates 6 publication-quality charts for 7-minute presentation.
All fixes applied:
- Chart 1: Fixed overlapping text
- Chart 3: 80M Lichess, removed unrealized rows (Training Time, Target ELO)
- Chart 4: One-liner phases (no hours), updated 80M
- Chart 5: Changed to "Hybrid Approach", simplified orange box
- Chart 6: Minimalist boxes, clean arrows, no metrics box
- Global: "YOUR" → "OUR" everywhere

Requirements:
    pip install matplotlib numpy

Usage:
    python generate_final_charts_corrected.py

Output:
    6 PNG files @ 300 DPI (publication quality)
'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Global configuration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

print("="*80)
print("GENERATING 6 CORRECTED CHARTS FOR 7-MINUTE PRESENTATION")
print("="*80)

# ============================================================================
# CHART 1: SLIDE 2 - Future Trajectory (FIXED OVERLAPPING TEXT)
# ============================================================================
print("\n[1/6] Generating Chart 1: Future Trajectory (FIXED)...")

fig, ax = plt.subplots(figsize=(12, 7))

markets = ['RL Market', 'AI Chess\nTraining', 'GPU Market']
values_2024 = [10.5, 0.15, 63.2]  # Billions USD
values_2032 = [88.7, 0.6, 592]    # Billions USD

x = np.arange(len(markets))
width = 0.35

bars1 = ax.bar(x - width/2, values_2024, width, label='2024', 
               color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.9)
bars2 = ax.bar(x + width/2, values_2032, width, label='2032', 
               color='#A23B72', edgecolor='black', linewidth=1.5, alpha=0.9)

# Add value labels on bars with NO OVERLAP
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height < 1:
            label = f'${height*1000:.0f}M'
            y_offset = 0.05  # Small offset for small bars
        else:
            label = f'${height:.1f}B'
            y_offset = max(height * 0.03, 5)  # Dynamic offset based on bar height

        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                label, ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('Market Size (USD)', fontsize=16, fontweight='bold')
ax.set_title('Three Markets Converging on Reinforcement Learning', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(markets, fontsize=15, fontweight='bold')
ax.legend(fontsize=14, loc='upper left', frameon=True, shadow=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 650)

# Add CAGR annotations with PROPER SPACING (NO OVERLAP)
# Position CAGR text AWAY from bars to prevent overlap
ax.text(0, 200, '41.5%\nCAGR', fontsize=13, ha='center', 
        color='#A23B72', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#A23B72', linewidth=2))

ax.text(1, 1.5, '20%\nCAGR', fontsize=13, ha='center', 
        color='#A23B72', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#A23B72', linewidth=2))

ax.text(2, 250, '28.2%\nCAGR', fontsize=13, ha='center', 
        color='#A23B72', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#A23B72', linewidth=2))

plt.tight_layout()
plt.savefig('chart1_future_trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart1_future_trajectory.png (overlapping text FIXED)")

# ============================================================================
# CHART 2: SLIDE 3 - Industry Leaders (no changes needed)
# ============================================================================
print("\n[2/6] Generating Chart 2: Industry Leaders...")

fig, ax = plt.subplots(figsize=(12, 7))

companies = ['Alphabet\n(Google)', 'Meta\nPlatforms', 'Tesla']
investments = [96.5, 39, 25.2]  # Billions USD
colors = ["#15FF00", '#0668E1', '#CC0000']

bars = ax.barh(companies, investments, color=colors, height=0.5, 
               edgecolor='black', linewidth=1.5, alpha=0.9)

# Add value labels and applications
applications = ['Gemini, AlphaZero, Waymo', 
                'Meta Superintelligence Labs',
                'Tesla\'s Full Self Driving']

for i, (bar, inv, app) in enumerate(zip(bars, investments, applications)):
    # Investment value
    ax.text(inv + 2, i, f'${inv:.1f}B', va='center', fontsize=14, fontweight='bold')
    # Application text
    ax.text(inv - 2, i, app, va='center', ha='right', fontsize=11, style='italic', color='black')

ax.set_xlabel('2024 Market Size (USD Billions)', fontsize=16, fontweight='bold')
ax.set_title('Industry Leaders in Reinforcement Learning based AI', 
             fontsize=18, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 110)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('chart2_industry_leaders.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart2_industry_leaders.png")

# ============================================================================
# CHART 3: SLIDE 4 - Comparison Table (UPDATED: 80M Lichess, removed rows)
# ============================================================================
print("\n[3/6] Generating Chart 3: Comparison Table (UPDATED)...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Table data - REMOVED Training Time and Target ELO rows, UPDATED to 80M Lichess
dimensions = ['Cost', 'Hardware', 'Data Source', 'Memory', 'MCTS Simulations']
alphazero = ['$25-35M', '5,000 TPUs', 'Pure self-play', 'N/A', '800/move']
our_impl = ['$2-5K', '3 NVIDIA GPUs\n(8GB RAM + 4GB GPU)', 
            '80M Lichess\nboard states\n+ self-play', 
            '320MB RAM\n170MB GPU', '200/move\n(4 parallel)']
advantage = ['15,000x\ncheaper', 'Commodity\nhardware', 'Hybrid\nconvergence', 
             '<1%\nusage', '4x\nthroughput']

# Create table
table_data = []
for i in range(len(dimensions)):
    table_data.append([dimensions[i], alphazero[i], our_impl[i], advantage[i]])

table = ax.table(cellText=table_data, 
                 colLabels=['Dimension', 'AlphaZero (2018)', 'Our Implementation', 'Advantage'],
                 cellLoc='left', loc='center', 
                 colWidths=[0.18, 0.24, 0.32, 0.26])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)

# Style data rows (alternating colors)
for i in range(1, len(dimensions) + 1):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F5F5F5')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_edgecolor('black')
        table[(i, j)].set_linewidth(1.5)

        # Bold the advantage column
        if j == 3:
            table[(i, j)].set_text_props(weight='bold', color='#A23B72')

ax.set_title('What Makes Our Approach Different', fontsize=20, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('chart3_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart3_comparison_table.png (80M Lichess, removed unrealized rows)")

# ============================================================================
# CHART 4: SLIDE 5 - Methodology Flowchart (ONE-LINERS, 80M, no hours)
# ============================================================================
print("\n[4/6] Generating Chart 4: Methodology Flowchart (ONE-LINERS)...")

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define phases with ONE-LINER format (no hours)
phases = [
    {
        'y': 7.5, 'color': '#2E86AB', 
        'title': 'Phase 1: Supervised Pretraining',
        'oneliner': '80M Lichess → ResNet (8 layers)'
    },
    {
        'y': 5, 'color': '#F18F01', 
        'title': 'Phase 2: MCTS Self-Play',
        'oneliner': '4 CPU workers, 200 MCTS/move'
    },
    {
        'y': 2.5, 'color': '#A23B72', 
        'title': 'Phase 3: Combined Fine-Tuning',
        'oneliner': '90% supervised + 10% self-play'
    }
]

# Draw boxes
for phase in phases:
    rect = mpatches.FancyBboxPatch((1, phase['y']), 8, 1.2,
                                   boxstyle="round,pad=0.1", 
                                   facecolor=phase['color'], 
                                   edgecolor='black', 
                                   linewidth=2.5, alpha=0.9)
    ax.add_patch(rect)

    # Title
    ax.text(5, phase['y'] + 0.95, phase['title'],
            fontsize=13, fontweight='bold', ha='center', va='top', color='white')

    # One-liner
    ax.text(5, phase['y'] + 0.45, phase['oneliner'],
            fontsize=12, ha='center', va='center', color='white')

# Draw arrows
arrow_props = dict(arrowstyle='->', lw=3, color='black')
ax.annotate('', xy=(5, 6.2), xytext=(5, 7.5), arrowprops=arrow_props)
ax.annotate('', xy=(5, 3.7), xytext=(5, 5), arrowprops=arrow_props)

# Add key decision annotation (KEEP)
ax.text(5, 0.8, 'Why CPU for MCTS?\nTree search is serial → GPU would be 2-3x SLOWER (tested)',
        fontsize=11, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.set_title('Technical Methodology: Hybrid Training Pipeline', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('chart4_methodology_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart4_methodology_flowchart.png (one-liners, 80M, no hours)")

# ============================================================================
# CHART 5: SLIDE 6 - Market Position (HYBRID APPROACH, simplified)
# ============================================================================
print("\n[5/6] Generating Chart 5: Market Position (HYBRID APPROACH)...")

fig, ax = plt.subplots(figsize=(11, 10))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Cost (Lower → Higher)', fontsize=16, fontweight='bold')
ax.set_ylabel('Approach Target', fontsize=16, fontweight='bold')
ax.set_xticks([2.5, 7.5])
ax.set_xticklabels(['Low Cost\n($2-5K)', 'High Cost\n($25M-75B)'], fontsize=14, fontweight='bold')
ax.set_yticks([2.5, 7.5])
ax.set_yticklabels(['Strong Club\nApproach', 'Superhuman\nApproach'], 
                   fontsize=14, fontweight='bold')

# Draw quadrant lines
ax.axvline(5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Plot companies
companies = [
    {'x': 8, 'y': 8, 'name': 'Google\n$75B', 'color': '#4285F4', 'size': 3500},
    {'x': 7, 'y': 7.5, 'name': 'Meta\n$60-72B', 'color': '#0668E1', 'size': 3000},
    {'x': 2, 'y': 3, 'name': 'HYBRID\nAPPROACH\n$2-5K', 'color': '#A23B72', 'size': 3500}
]

for comp in companies:
    ax.scatter(comp['x'], comp['y'], s=comp['size'], c=comp['color'], 
               edgecolors='black', linewidths=2.5, alpha=0.85, zorder=3)
    ax.text(comp['x'], comp['y'], comp['name'], fontsize=12, fontweight='bold',
            ha='center', va='center', color='white', zorder=4)

# Simplified orange box (Option B style)
ax.text(2, 1.2, 'Chess = Proof of Concept\nTransfers to: Robotics, Game AI, Academic Research',
        fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#F18F01', alpha=0.8, edgecolor='black', linewidth=2))

# TAM box
ax.text(7.5, 1.2, 'Target: $38B TAM\n• Mid-Market: $30B\n• Startups: $8.7B',
        fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=1.5))

ax.set_title('Market Position: Methodology Validation via Chess', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('chart5_market_position.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart5_market_position.png (Hybrid Approach, simplified)")

# ============================================================================
# CHART 6: BONUS - Architecture Diagram (MINIMALIST, clean arrows)
# ============================================================================
print("\n[6/6] Generating Chart 6: Architecture Diagram (MINIMALIST)...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# CNN box at top - MINIMALIST
cnn_box = mpatches.FancyBboxPatch((4, 7), 4, 1, boxstyle="round,pad=0.1", 
                                   facecolor='#2E86AB', edgecolor='black', 
                                   linewidth=2.5, alpha=0.9)
ax.add_patch(cnn_box)
ax.text(6, 7.5, 'Trained CNN (120 MB)', fontsize=13, fontweight='bold', ha='center', color='white')

# GPU Server box - MINIMALIST
gpu_box = mpatches.FancyBboxPatch((3.5, 4.5), 5, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor='#F18F01', edgecolor='black', 
                                   linewidth=2.5, alpha=0.9)
ax.add_patch(gpu_box)
ax.text(6, 5.1, 'GPU Server (Batched 16)', fontsize=13, fontweight='bold', ha='center', color='white')

# CPU Worker boxes - MINIMALIST
worker_positions = [(1.5, 2), (4, 2), (6.5, 2), (9, 2)]
for i, (x, y) in enumerate(worker_positions):
    worker_box = mpatches.FancyBboxPatch((x, y), 1.5, 1.2, boxstyle="round,pad=0.05", 
                                          facecolor='#A23B72', edgecolor='black', 
                                          linewidth=2, alpha=0.9)
    ax.add_patch(worker_box)
    ax.text(x + 0.75, y + 0.6, f'CPU W{i}\n(200 MCTS)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')

# CLEAN ARROWS - Single lines, thicker, fewer
arrow_props_thick = dict(arrowstyle='->', lw=3, color='black')

# CNN to GPU (single clean arrow)
ax.annotate('', xy=(6, 5.7), xytext=(6, 7), arrowprops=arrow_props_thick)

# GPU to Workers (clean, simple lines)
for x, y in worker_positions:
    ax.annotate('', xy=(x + 0.75, y + 1.2), xytext=(6, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# NO METRICS BOX (removed as requested)

ax.set_title('Parallel Architecture: 4 CPU Workers + 1 GPU Server', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('chart6_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: chart6_architecture_diagram.png (minimalist, clean arrows)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL 6 CORRECTED CHARTS GENERATED SUCCESSFULLY")
print("="*80)
print("\nGenerated files:")
print("  1. chart1_future_trajectory.png       (FIXED overlapping text)")
print("  2. chart2_industry_leaders.png        (no changes)")
print("  3. chart3_comparison_table.png        (80M Lichess, removed unrealized rows)")
print("  4. chart4_methodology_flowchart.png   (one-liners, 80M, no hours)")
print("  5. chart5_market_position.png         (Hybrid Approach, simplified)")
print("  6. chart6_architecture_diagram.png    (minimalist, clean arrows)")
print("\nAll corrections applied:")
print("  ✓ Chart 1: Fixed CAGR annotation overlaps")
print("  ✓ Chart 3: 80M Lichess, removed Training Time & Target ELO rows")
print("  ✓ Chart 4: One-liner phases, 80M updated, no hours")
print("  ✓ Chart 5: 'Hybrid Approach', simplified orange box")
print("  ✓ Chart 6: Minimalist boxes, clean arrows, no metrics")
print("  ✓ Global: 'YOUR' → 'OUR' everywhere")
print("="*80)
