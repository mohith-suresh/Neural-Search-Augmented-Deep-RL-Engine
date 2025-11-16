#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from math import pi

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

print("="*80)
print("Generating 5 publication-quality charts")
print("="*80)

# ============================================================================
# CHART 1: Revenue Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

companies = ['NVIDIA Data\nCenter Q3 FY25', 'Alphabet\nQ4 2024', 
             'Tesla\nQ3 2024', 'Apple\nR&D FY24', 'Meta CapEx\n2024']
revenues = [30.8, 96.5, 25.2, 31.4, 40.0]
colors = ['#76B900', '#4285F4', '#CC0000', '#555555', '#0668E1']

bars = ax.barh(companies, revenues, color=colors, height=0.65, 
               edgecolor='black', linewidth=1.5)

for i, (bar, rev) in enumerate(zip(bars, revenues)):
    ax.text(rev + 2.5, i, f'${rev}B', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Revenue/Investment (USD Billions)', fontsize=12, fontweight='bold')
ax.set_title('2024 Verified Financial Metrics: Top 5 Companies\n(From SEC Filings & Official Reports)', 
             fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 108)
ax.grid(axis='x', alpha=0.3, linestyle='--')

fig.text(0.98, 0.01, 'Sources: SEC filings (Alphabet, NVIDIA, Tesla, Apple), Meta earnings', 
         ha='right', fontsize=7, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('chart1_verified_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1 saved")

# ============================================================================
# CHART 2: Market Growth
# ============================================================================
fig, ax = plt.subplots(figsize=(13, 7))

markets = ['RL', 'GPU\nMarket', 'Autonomous\nVehicles', 'AI\nRobotics', 'Drug\nDiscovery']
values_2024 = [10.5, 63.2, 1.9, 6.1, 1.5]
values_2032 = [88.7, 592, 43.8, 33.4, 15.3]

x = np.arange(len(markets))
width = 0.35

bars1 = ax.bar(x - width/2, values_2024, width, label='2024', 
               color='#5B9BD5', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, values_2032, width, label='2030-2032', 
               color='#ED7D31', edgecolor='black', linewidth=1.2)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 8,
                f'${height:.1f}B', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Market Size (USD Billions)', fontsize=12, fontweight='bold')
ax.set_title('Market Growth: AlphaZero-Inspired RL Applications', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(markets, fontsize=11)
ax.legend(fontsize=11, loc='upper left', frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 650)

fig.text(0.98, 0.01, 'Sources: Allied MR, Fortune BI, Grand View, PRNewswire, Global Insight', 
         ha='right', fontsize=7, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('chart2_verified_market_growth.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved")

# ============================================================================
# CHART 3: CAGR - LEGEND MOVED TO TOP RIGHT
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6.5))

markets_cagr = ['Autonomous Vehicles', 'AI Chip Market', 'Reinforcement Learning', 
                'AI Robotics', 'Data Center GPU', 'GPU Market', 'Drug Discovery']
cagrs = [73.5, 44.9, 41.5, 40.2, 28.5, 28.2, 26.1]
colors_cagr = ['#C00000' if c > 40 else '#ED7D31' if c > 30 else '#FFC000' for c in cagrs]

bars = ax.barh(markets_cagr, cagrs, color=colors_cagr, height=0.6, 
               edgecolor='black', linewidth=1.2)

for bar, cagr in zip(bars, cagrs):
    ax.text(cagr + 1.2, bar.get_y() + bar.get_height()/2, f'{cagr}%', 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Compound Annual Growth Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Growth Rate Comparison: Fastest Growing Markets', 
             fontsize=13, fontweight='bold', pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 80)
ax.grid(axis='x', alpha=0.3, linestyle='--')

red_patch = mpatches.Patch(color='#C00000', label='High (>40%)')
orange_patch = mpatches.Patch(color='#ED7D31', label='Medium (30-40%)')
yellow_patch = mpatches.Patch(color='#FFC000', label='Steady (<30%)')
ax.legend(handles=[red_patch, orange_patch, yellow_patch], fontsize=9.5, 
          loc='upper right', frameon=False)

fig.text(0.98, 0.01, 'Sources: Market research firms (Allied MR, Fortune BI, Grand View, Precedence)', 
         ha='right', fontsize=7, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.025, 1, 1])
plt.savefig('chart3_verified_cagr.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3 saved")

# ============================================================================
# CHART 4: Timeline
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

years = [2016, 2018, 2020, 2021, 2024, 2024, 2024, 2025, 2025]
companies_timeline = ['Google', 'Google', 'Google', 'Google', 
                      'Meta', 'NVIDIA', 'Tesla', 'Apple', 'Meta']
achievements = ['Data Center\nCooling -40%', 'AlphaZero\n(Science)', 
                'MuZero\n(Nature)', 'AlphaFold 2\n(Nature)',
                'Meta Motivo\nSOTA RL', 'H100\n+112% YoY', 
                'FSD 2B+\nmiles', 'GIGAFLOW\n(ICML)', 
                'CapEx\n$60-72B']
company_colors = {'Google': '#4285F4', 'Meta': '#0668E1', 'NVIDIA': '#76B900', 
                  'Tesla': '#CC0000', 'Apple': '#555555'}

for i, (year, company, achievement) in enumerate(zip(years, companies_timeline, achievements)):
    color = company_colors[company]
    y_pos = i * 1.1

    ax.plot([year-0.25, year+0.25], [y_pos, y_pos], color=color, linewidth=4.5, alpha=0.8)
    ax.scatter([year], [y_pos], s=200, color=color, edgecolors='black', 
               linewidths=2, zorder=3, alpha=0.95)

    ax.text(year + 0.5, y_pos, achievement, va='center', fontsize=9.5, 
            fontweight='bold', color=color)
    ax.text(year - 0.5, y_pos, str(year), va='center', ha='right', 
            fontsize=8.5, color='gray', fontweight='bold')

ax.set_xlim(2015.5, 2025.5)
ax.set_ylim(-0.8, len(years) * 1.1)
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_title('Timeline: AlphaZero-Inspired Breakthroughs (2016-2025)', 
             fontsize=13, fontweight='bold', pad=15)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([])
ax.grid(axis='x', alpha=0.2, linestyle='--')

legend_elements = [mpatches.Patch(facecolor=color, label=company, edgecolor='black') 
                   for company, color in company_colors.items()]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left', ncol=5, frameon=False)

fig.text(0.98, 0.01, 'Sources: Science 2018, Nature 2020/2021, ICML 2025, SEC filings', 
         ha='right', fontsize=7, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig('chart4_verified_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4 saved")

# ============================================================================
# CHART 5: Radar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Research\nImpact', 'Commercial\nDeploy', 'Open\nSource', 
              'Hardware', 'Market\nSize', 'Innovation\nSpeed']
N = len(categories)

companies_radar = ['Google DeepMind', 'Meta AI', 'NVIDIA', 'Tesla', 'Apple']
values = {
    'Google DeepMind': [10, 8, 7, 4, 6, 9],
    'Meta AI': [8, 5, 10, 3, 4, 7],
    'NVIDIA': [6, 10, 6, 10, 10, 8],
    'Tesla': [5, 7, 2, 7, 8, 7],
    'Apple': [7, 6, 8, 6, 7, 8]
}
colors_radar = ['#4285F4', '#0668E1', '#76B900', '#CC0000', '#555555']

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for i, (company, color) in enumerate(zip(companies_radar, colors_radar)):
    company_values = values[company] + values[company][:1]
    ax.plot(angles, company_values, 'o-', linewidth=2.3, label=company, 
            color=color, markersize=7)
    ax.fill(angles, company_values, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10.5)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, color='gray')
ax.set_title('Multi-Dimensional Company Comparison\n(Relative Scores 0-10)', 
             fontsize=13, fontweight='bold', pad=25)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.08), fontsize=10, frameon=False)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('chart5_verified_radar.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5 saved")

print("\n" + "="*80)
print("All 5 charts generated successfully")
print("="*80)
print("Changes made:")
print("  - Chart 6 (Project Connections) REMOVED")
print("  - Chart 3 legend moved from 'lower right' to 'upper right'")
print("  - Now generating 5 charts total")
print("="*80)
