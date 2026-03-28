"""
Generate all figures for the CS 4452 Final Report.
Outputs saved to figures/ directory.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Create output directory ──
os.makedirs('figures', exist_ok=True)

# ── Color palette ──
MULTI_COLOR = '#2563EB'   # blue
TEXT_COLOR  = '#059669'    # green
IMAGE_COLOR = '#DC2626'   # red
BG_COLOR    = '#FAFAFA'
GRID_COLOR  = '#E5E7EB'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.facecolor': BG_COLOR,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

# ══════════════════════════════════════════════════════════════
# DATA (from results/*.csv)
# ══════════════════════════════════════════════════════════════

epochs = [1, 2, 3, 4, 5]

# Multimodal
mm_train_loss = [0.637, 0.520, 0.397, 0.262, 0.156]
mm_train_acc  = [0.643, 0.748, 0.823, 0.900, 0.950]
mm_val_acc    = [0.693, 0.725, 0.725, 0.717, 0.712]
mm_train_f1   = [0.422, 0.659, 0.772, 0.875, 0.937]
mm_val_f1     = [0.619, 0.595, 0.657, 0.637, 0.616]
mm_train_auc  = [0.650, 0.808, 0.897, 0.957, 0.986]
mm_val_auc    = [0.754, 0.773, 0.791, 0.778, 0.768]

# Text-only
txt_train_loss = [0.643, 0.578, 0.514, 0.455, 0.412]
txt_train_acc  = [0.643, 0.708, 0.747, 0.779, 0.801]
txt_val_acc    = [0.689, 0.688, 0.676, 0.657, 0.649]
txt_train_f1   = [0.385, 0.583, 0.660, 0.713, 0.750]
txt_val_f1     = [0.460, 0.567, 0.521, 0.527, 0.551]
txt_train_auc  = [0.639, 0.747, 0.814, 0.859, 0.886]
txt_val_auc    = [0.725, 0.733, 0.721, 0.711, 0.710]

# Image-only
img_train_loss = [0.673, 0.611, 0.501, 0.356, 0.236]
img_train_acc  = [0.589, 0.669, 0.763, 0.852, 0.917]
img_val_acc    = [0.603, 0.648, 0.630, 0.630, 0.625]
img_train_f1   = [0.229, 0.484, 0.673, 0.804, 0.893]
img_val_f1     = [0.132, 0.490, 0.444, 0.514, 0.519]
img_train_auc  = [0.551, 0.710, 0.840, 0.932, 0.976]
img_val_auc    = [0.637, 0.662, 0.653, 0.655, 0.652]

# Test results
test_acc = {'Multimodal': 0.726, 'Text-only': 0.710, 'Image-only': 0.621}
test_f1  = {'Multimodal': 0.669, 'Text-only': 0.617, 'Image-only': 0.450}
test_auc = {'Multimodal': 0.788, 'Text-only': 0.757, 'Image-only': 0.620}


# ══════════════════════════════════════════════════════════════
# FIGURE 1 – Training Loss Curves (all 3 models)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(epochs, mm_train_loss,  'o-', color=MULTI_COLOR, linewidth=2.5, markersize=8, label='Multimodal')
ax.plot(epochs, txt_train_loss, 's-', color=TEXT_COLOR,  linewidth=2.5, markersize=8, label='Text-only')
ax.plot(epochs, img_train_loss, '^-', color=IMAGE_COLOR, linewidth=2.5, markersize=8, label='Image-only')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Training Loss', fontsize=13)
ax.set_title('Figure 1: Training Loss Curves', fontsize=15, fontweight='bold', pad=12)
ax.set_xticks(epochs)
ax.legend(fontsize=12, framealpha=0.9)
ax.set_ylim(0, 0.75)
plt.tight_layout()
plt.savefig('figures/fig1_training_loss.png', dpi=200)
plt.close()
print("✓ Figure 1 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 – Train vs Val Accuracy (all 3 models, subplots)
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (name, tr, va, color) in zip(axes, [
    ('Multimodal', mm_train_acc, mm_val_acc, MULTI_COLOR),
    ('Text-only',  txt_train_acc, txt_val_acc, TEXT_COLOR),
    ('Image-only', img_train_acc, img_val_acc, IMAGE_COLOR),
]):
    ax.plot(epochs, tr, 'o-', color=color, linewidth=2.5, markersize=8, label='Train')
    ax.plot(epochs, va, 's--', color=color, linewidth=2.5, markersize=8, alpha=0.6, label='Validation')
    ax.fill_between(epochs, tr, va, color=color, alpha=0.08)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 1.0)

axes[0].set_ylabel('Accuracy', fontsize=13)
fig.suptitle('Figure 2: Training vs. Validation Accuracy', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig2_accuracy_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 – Validation AUC Curves (all 3 models)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(epochs, mm_val_auc,  'o-', color=MULTI_COLOR, linewidth=2.5, markersize=8, label='Multimodal')
ax.plot(epochs, txt_val_auc, 's-', color=TEXT_COLOR,  linewidth=2.5, markersize=8, label='Text-only')
ax.plot(epochs, img_val_auc, '^-', color=IMAGE_COLOR, linewidth=2.5, markersize=8, label='Image-only')

# Mark best epoch for each
best_mm = epochs[np.argmax(mm_val_auc)]
best_txt = epochs[np.argmax(txt_val_auc)]
best_img = epochs[np.argmax(img_val_auc)]
ax.axvline(x=best_mm, color=MULTI_COLOR, linestyle=':', alpha=0.4)
ax.axvline(x=best_txt, color=TEXT_COLOR, linestyle=':', alpha=0.4)

ax.annotate(f'Best: {max(mm_val_auc):.3f}', xy=(best_mm, max(mm_val_auc)),
            xytext=(best_mm+0.3, max(mm_val_auc)+0.015), fontsize=10, color=MULTI_COLOR,
            arrowprops=dict(arrowstyle='->', color=MULTI_COLOR, lw=1.2))

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Validation ROC-AUC', fontsize=13)
ax.set_title('Figure 3: Validation ROC-AUC per Epoch', fontsize=15, fontweight='bold', pad=12)
ax.set_xticks(epochs)
ax.legend(fontsize=12, framealpha=0.9)
ax.set_ylim(0.6, 0.85)
plt.tight_layout()
plt.savefig('figures/fig3_val_auc_curves.png', dpi=200)
plt.close()
print("✓ Figure 3 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 – Test Set Performance Bar Chart
# ══════════════════════════════════════════════════════════════

models = ['Multimodal\n(RoBERTa + ViT)', 'Text-only\n(RoBERTa)', 'Image-only\n(ViT)']
colors = [MULTI_COLOR, TEXT_COLOR, IMAGE_COLOR]

x = np.arange(len(models))
width = 0.22

fig, ax = plt.subplots(figsize=(10, 6))

acc_vals = list(test_acc.values())
f1_vals  = list(test_f1.values())
auc_vals = list(test_auc.values())

bars1 = ax.bar(x - width, acc_vals, width, label='Accuracy', color='#3B82F6', edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x,         f1_vals,  width, label='F1-Score', color='#10B981', edgecolor='white', linewidth=0.8)
bars3 = ax.bar(x + width, auc_vals, width, label='ROC-AUC',  color='#F59E0B', edgecolor='white', linewidth=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Figure 4: Test Set Performance Comparison', fontsize=15, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.set_ylim(0, 0.95)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random baseline')
plt.tight_layout()
plt.savefig('figures/fig4_test_performance.png', dpi=200)
plt.close()
print("✓ Figure 4 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 5 – Overfitting Gap (Train-Val Accuracy Difference)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5.5))
mm_gap  = [t - v for t, v in zip(mm_train_acc, mm_val_acc)]
txt_gap = [t - v for t, v in zip(txt_train_acc, txt_val_acc)]
img_gap = [t - v for t, v in zip(img_train_acc, img_val_acc)]

ax.plot(epochs, mm_gap,  'o-', color=MULTI_COLOR, linewidth=2.5, markersize=8, label='Multimodal')
ax.plot(epochs, txt_gap, 's-', color=TEXT_COLOR,  linewidth=2.5, markersize=8, label='Text-only')
ax.plot(epochs, img_gap, '^-', color=IMAGE_COLOR, linewidth=2.5, markersize=8, label='Image-only')

ax.fill_between(epochs, 0, img_gap, color=IMAGE_COLOR, alpha=0.07)
ax.axhline(y=0, color='gray', linewidth=1)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Train Acc − Val Acc (Overfit Gap)', fontsize=13)
ax.set_title('Figure 5: Overfitting Analysis', fontsize=15, fontweight='bold', pad=12)
ax.set_xticks(epochs)
ax.legend(fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig('figures/fig5_overfitting_gap.png', dpi=200)
plt.close()
print("✓ Figure 5 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 6 – Dataset Class Distribution Pie Chart
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 7))
sizes = [6931, 4725]
labels_pie = ['Not Hateful\n6,931 (59.5%)', 'Hateful\n4,725 (40.5%)']
colors_pie = ['#60A5FA', '#F87171']
explode = (0.03, 0.03)

wedges, texts = ax.pie(sizes, labels=labels_pie, colors=colors_pie, explode=explode,
                       startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Figure 6: Dataset Class Distribution (N = 11,656)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig6_class_distribution.png', dpi=200)
plt.close()
print("✓ Figure 6 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 7 – Data Split Breakdown Bar
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 2.5))
splits = ['Train (70%)', 'Validation (10%)', 'Test (20%)']
counts = [8159, 1165, 2332]
bar_colors = ['#3B82F6', '#10B981', '#F59E0B']

bars = ax.barh(splits, counts, color=bar_colors, edgecolor='white', height=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', fontsize=13, fontweight='bold')

ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_title('Figure 7: Dataset Split Sizes', fontsize=14, fontweight='bold', pad=10)
ax.set_xlim(0, 10000)
plt.tight_layout()
plt.savefig('figures/fig7_data_splits.png', dpi=200)
plt.close()
print("✓ Figure 7 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 8 – System Architecture Diagram (text-based)
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=11):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor='#374151', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white' if color != '#FEF3C7' else '#92400E')

# Input
draw_box(ax, 0.5, 4.5, 2.2, 1.2, 'Meme Image\n(224×224 PNG)', '#DBEAFE', fontsize=10)
draw_box(ax, 0.5, 1.8, 2.2, 1.2, 'Meme Text\n(OCR extracted)', '#D1FAE5', fontsize=10)

# Encoders
draw_box(ax, 4, 4.5, 2.5, 1.2, 'ViT-base\n(Image Encoder)', IMAGE_COLOR)
draw_box(ax, 4, 1.8, 2.5, 1.2, 'RoBERTa-base\n(Text Encoder)', TEXT_COLOR)

# Features
draw_box(ax, 7.8, 4.5, 1.8, 1.2, 'Image\nFeatures\n(768-d)', '#FEF3C7')
draw_box(ax, 7.8, 1.8, 1.8, 1.2, 'Text\nFeatures\n(768-d)', '#FEF3C7')

# Concatenation
draw_box(ax, 10.5, 3, 1.8, 1.5, 'Concat\n+\nMLP\n(1536→512→1)', MULTI_COLOR)

# Output
draw_box(ax, 12.8, 3.2, 1, 1, 'Hateful?\n0/1', '#7C3AED')

# Arrows
arrow_style = dict(arrowstyle='->', color='#6B7280', lw=2)
ax.annotate('', xy=(4, 5.1), xytext=(2.7, 5.1), arrowprops=arrow_style)
ax.annotate('', xy=(4, 2.4), xytext=(2.7, 2.4), arrowprops=arrow_style)
ax.annotate('', xy=(7.8, 5.1), xytext=(6.5, 5.1), arrowprops=arrow_style)
ax.annotate('', xy=(7.8, 2.4), xytext=(6.5, 2.4), arrowprops=arrow_style)
ax.annotate('', xy=(10.5, 4.0), xytext=(9.6, 5.0), arrowprops=arrow_style)
ax.annotate('', xy=(10.5, 3.5), xytext=(9.6, 2.5), arrowprops=arrow_style)
ax.annotate('', xy=(12.8, 3.7), xytext=(12.3, 3.75), arrowprops=arrow_style)

ax.set_title('Figure 8: Multimodal Fusion Architecture', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig8_architecture.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 8 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 9 – Training F1-Score Curves
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for ax, (name, tr, va, color) in zip(axes, [
    ('Multimodal', mm_train_f1, mm_val_f1, MULTI_COLOR),
    ('Text-only',  txt_train_f1, txt_val_f1, TEXT_COLOR),
    ('Image-only', img_train_f1, img_val_f1, IMAGE_COLOR),
]):
    ax.plot(epochs, tr, 'o-', color=color, linewidth=2.5, markersize=8, label='Train')
    ax.plot(epochs, va, 's--', color=color, linewidth=2.5, markersize=8, alpha=0.6, label='Validation')
    ax.fill_between(epochs, tr, va, color=color, alpha=0.08)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

axes[0].set_ylabel('F1-Score', fontsize=13)
fig.suptitle('Figure 9: Training vs. Validation F1-Score', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig9_f1_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 9 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 10 – Modality Contribution Radar/Spider Chart
# ══════════════════════════════════════════════════════════════

categories = ['Accuracy', 'F1-Score', 'ROC-AUC']
N = len(categories)

vals_mm  = [0.726, 0.669, 0.788]
vals_txt = [0.710, 0.617, 0.757]
vals_img = [0.621, 0.450, 0.620]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for vals, color, label in [
    (vals_mm, MULTI_COLOR, 'Multimodal'),
    (vals_txt, TEXT_COLOR, 'Text-only'),
    (vals_img, IMAGE_COLOR, 'Image-only'),
]:
    values = vals + vals[:1]
    ax.plot(angles, values, 'o-', color=color, linewidth=2.5, markersize=8, label=label)
    ax.fill(angles, values, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0.3, 0.85)
ax.set_title('Figure 10: Model Comparison Radar', fontsize=15, fontweight='bold', pad=25)
ax.legend(loc='lower right', fontsize=11, bbox_to_anchor=(1.25, -0.05))
plt.tight_layout()
plt.savefig('figures/fig10_radar_chart.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Figure 10 saved")

print("\n✅ All 10 figures saved to figures/ directory!")
