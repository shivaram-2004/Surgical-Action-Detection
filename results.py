"""
MESAD — Publication-ready figures
Run: pip install matplotlib numpy
     python plot_results.py
Outputs: fig1_progression.pdf, fig2_learning_curve.pdf,
         fig3_per_class_map.pdf, fig4_precision_recall.pdf
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linewidth'   : 0.5,
})

BLUE   = '#2563EB'
GREEN  = '#059669'
RED    = '#DC2626'
AMBER  = '#D97706'
TEAL   = '#0D9488'
LGRAY  = '#E5E7EB'

# ─────────────────────────────────────────────
# FIG 1 — Model progression
# ─────────────────────────────────────────────
steps  = ['Step 1\nYOLOv8n\nPhantom',
          'Step 2\nYOLOv8s\nPhantom',
          'Step 3\nYOLOv8s\nReal',
          'Step 4\nYOLOv8s\nCombined']
map50  = [0.208, 0.187, 0.110, 0.146]
map595 = [0.0376, 0.0635, 0.0336, 0.0461]

x   = np.arange(len(steps))
w   = 0.35
fig, ax = plt.subplots(figsize=(7, 3.8))
b1 = ax.bar(x - w/2, map50,  w, label='mAP@0.5', color=BLUE,  alpha=0.9, zorder=3)
b2 = ax.bar(x + w/2, map595, w, label='mAP@0.5:0.95', color=TEAL,  alpha=0.9, zorder=3)

for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=BLUE)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8, color=TEAL)

ax.set_xticks(x)
ax.set_xticklabels(steps, linespacing=1.4)
ax.set_ylabel('Detection score')
ax.set_ylim(0, 0.28)
ax.legend(framealpha=0.7, loc='upper right')
ax.set_title('Fig. 2 — mAP across training steps (MESAD dataset)')
plt.tight_layout()
plt.savefig('fig2_progression.pdf')
plt.savefig('fig3_progression.png')
print("✅ fig2_progression saved")
plt.close()

# ─────────────────────────────────────────────
# FIG 3 — Step 5 learning curve
# ─────────────────────────────────────────────
epochs     = np.arange(1, 9)
train_acc  = [30.0, 38.1, 43.4, 49.2, 54.2, 59.0, 63.5, 67.4]
val_acc    = [33.3, 38.2, 38.2, 30.9, 38.2, 38.2, 39.7, 38.2]
train_loss = [2.456, 2.255, 2.076, 1.978, 1.817, 1.720, 1.590, 1.470]

fig, ax1 = plt.subplots(figsize=(7, 3.8))
ax2 = ax1.twinx()
ax2.spines['right'].set_visible(True)

l1, = ax1.plot(epochs, train_acc,  '-o', color=BLUE,  lw=2, ms=5, label='Train accuracy')
l2, = ax1.plot(epochs, val_acc,    '-s', color=GREEN, lw=2, ms=5, label='Val accuracy')
l3, = ax2.plot(epochs, train_loss, '--^',color=RED,   lw=1.5, ms=4, label='Train loss', alpha=0.8)

ax1.axvline(x=5, color=AMBER, lw=1.2, ls=':', alpha=0.9)
ax1.text(5.1, 15, 'YOLO\nunfreeze', fontsize=8, color=AMBER, va='bottom')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 80)
ax1.set_xticks(epochs)
ax2.set_ylabel('Training loss', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(1.2, 2.8)

lines = [l1, l2, l3]
ax1.legend(lines, [l.get_label() for l in lines], framealpha=0.7, loc='upper left')
ax1.set_title('Fig. 3 — Step 5 learning curve (multi-stream model, 8 epochs)')
plt.tight_layout()
plt.savefig('fig2_learning_curve.pdf')
plt.savefig('fig3_learning_curve.png')
print("✅ fig2_learning_curve saved")
plt.close()

# ─────────────────────────────────────────────
# FIG 3 — Per-class mAP@0.5
# ─────────────────────────────────────────────
classes = [
    'MovingDownBladder','GraspingCatheter','BladderAnastomosis',
    'SuckingBlood','SuckingSmoke','PullingSeminalVesicle',
    'CuttingTissue','PullingProstate','CuttingMesocolon',
    'ClippingTissue','CuttingVasDeferens','PassingNeedle',
    'CuttingSeminalVesicle','PullingTissue','CuttingThread',
    'ClippingBladderNeck','ClippingVasDeferens','PullingVasDeferens',
    'BaggingProstate','UrethraDissection','BladderNeckDissection',
    'CuttingProstate','PullingBladderNeck','ClippingSeminalVesicle'
]
map50_cls = [
    0.597,0.457,0.254,0.309,0.271,0.227,0.180,0.176,0.122,
    0.098,0.070,0.068,0.083,0.123,0.054,0.064,0.070,0.050,
    0.058,0.098,0.034,0.011,0.027,0.000
]
avg_map = np.mean(map50_cls)
colors  = [BLUE if v >= avg_map else RED for v in map50_cls]

fig, ax = plt.subplots(figsize=(7, 9))
y = np.arange(len(classes))
bars = ax.barh(y, map50_cls, color=colors, alpha=0.88, zorder=3, height=0.7)
ax.axvline(x=avg_map, color='gray', lw=1, ls='--', alpha=0.6, label=f'Mean ({avg_map:.3f})')

for i, (bar, val) in enumerate(zip(bars, map50_cls)):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=7.5,
            color=BLUE if val >= avg_map else RED)

ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=8.5)
ax.set_xlabel('mAP@0.5')
ax.set_xlim(0, 0.72)
ax.invert_yaxis()
p1 = mpatches.Patch(color=BLUE, label='Above mean', alpha=0.88)
p2 = mpatches.Patch(color=RED,  label='Below mean', alpha=0.88)
ax.legend(handles=[p1, p2, plt.Line2D([0],[0],color='gray',ls='--',lw=1,label=f'Mean ({avg_map:.3f})')],
          framealpha=0.7, fontsize=8)
ax.set_title('Fig. 4 — Per-class mAP@0.5, YOLOv8s combined dataset (Step 4)')
plt.tight_layout()
plt.savefig('fig3_per_class_map.pdf')
plt.savefig('fig4_per_class_map.png')
print("✅ fig3_per_class_map saved")
plt.close()

# ─────────────────────────────────────────────
# FIG 4 — Precision vs Recall
# ─────────────────────────────────────────────
precision = [0.508,0.360,0.281,0.347,0.432,0.343,0.281,0.231,0.383,
             0.159,0.455,0.121,0.119,0.241,0.164,0.217,0.941,0.139,
             0.000,0.219,0.100,0.011,0.012,1.000]
recall    = [0.765,0.531,0.439,0.477,0.293,0.427,0.297,0.278,0.125,
             0.375,0.037,0.044,0.146,0.209,0.118,0.125,0.067,0.096,
             0.000,0.113,0.032,0.231,0.005,0.000]

fig, ax = plt.subplots(figsize=(7, 9))
y  = np.arange(len(classes))
w2 = 0.32
ax.barh(y + w2/2, precision, w2, color=BLUE,  alpha=0.88, label='Precision', zorder=3)
ax.barh(y - w2/2, recall,    w2, color=GREEN, alpha=0.88, label='Recall',    zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(classes, fontsize=8.5)
ax.set_xlabel('Score')
ax.set_xlim(0, 1.1)
ax.invert_yaxis()
ax.legend(framealpha=0.7, fontsize=8)
ax.set_title('Fig. 5 — Precision vs recall per class, YOLOv8s combined (Step 4)')
plt.tight_layout()
plt.savefig('fig5_precision_recall.pdf')
plt.savefig('fig6_precision_recall.png')
print("✅ fig5_precision_recall saved")
plt.close()

print("\n🎉 All figures saved!")
print("   fig1_progression.pdf/png")
print("   fig2_learning_curve.pdf/png")
print("   fig3_per_class_map.pdf/png")
print("   fig4_precision_recall.pdf/png")
print("   fig5_precision_recall.pdf/png")
print("   fig6_precision_recall.png")
print("\n   Use PDF versions for IEEE paper submission (vector quality)")
print("   Use PNG versions for slides and faculty presentation")