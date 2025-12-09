"""
Crash version of moon_test.py
Same structure, same threshold rules, same spans, same DP workflow,
but using crash longitude/latitude instead of make_moons().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import noisy_grid_to_spans as ns   # same module as moon_test.py

# ================================================================
# 1. LOAD CRASH DATA (lon, lat)
# ================================================================
CSV_PATH = r"D:/Study_spare/FSU/Work/Database/group project/COP5725_group_project/crashes_240928/crashes_240928.csv"
XCOL = "LONGITUDE"
YCOL = "LATITUDE"

def load_crash_xy(path):
    df = pd.read_csv(path, usecols=[XCOL, YCOL])
    df = df.dropna(subset=[XCOL, YCOL])
    X = df[[XCOL, YCOL]].to_numpy(dtype=float)
    return X

X = load_crash_xy(CSV_PATH)
print(f"[Crash] Loaded {len(X)} points")

# ================================================================
# 2. GRID + COUNTS  (same style as moon_test.py)
# ================================================================
def make_grid(points, xlim, ylim, n_x, n_y):
    xs = np.linspace(xlim[0], xlim[1], n_x + 1)
    ys = np.linspace(ylim[0], ylim[1], n_y + 1)
    xi = np.clip(np.digitize(points[:, 0], xs) - 1, 0, n_x - 1)
    yi = np.clip(np.digitize(points[:, 1], ys) - 1, 0, n_y - 1)
    H = np.zeros((n_x, n_y), dtype=float)
    np.add.at(H, (xi, yi), 1)
    return H, xs, ys

# GRID_N = 1940   # you can choose 28, 64, 128 like in moon_test
GRID_N = 64   # you can choose 28, 64, 128 like in moon_test
xlim = (X[:,0].min(), X[:,0].max())
ylim = (X[:,1].min(), X[:,1].max())
H, xs, ys = make_grid(X, xlim, ylim, GRID_N, GRID_N)
print("Grid built:", H.shape)

# ================================================================
# 3. LAPLACE NOISE (same code as moon_test.py)
# ================================================================
def laplace_mech(counts, epsilon, clip_zero=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    b = 1.0 / float(epsilon)
    noise = rng.laplace(loc=0.0, scale=b, size=counts.shape)
    noisy = counts + noise
    if clip_zero:
        noisy = np.maximum(noisy, 0.0)
    return noisy, noise

eps_priv = 1.0
noisy1, _ = laplace_mech(H, eps_priv, clip_zero=True)

# ================================================================
# 4. THRESHOLD T (EXACT SAME RULE as moon_test.py)  !!!
# ================================================================
num_grids = H.size
n = len(X)          
print(f"Crash n ={n})")         # 1,860,785 in paper, here 1872846
def T_from_repo_rule(epsilon, num_grids, n):
    return (1.0 / epsilon) * np.log(num_grids / float(n))

# initial rule
if num_grids > 2 * n:
    T = T_from_repo_rule(eps_priv, num_grids, n)
else:
    T = None   # fallback required

# Fallback helpers (copied exactly from moon_test.py)
def estimate_kappa(alpha, cw_x, cw_y):
    r = int(np.ceil(max(alpha/cw_x, alpha/cw_y)))
    k = 0
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            if (dx*cw_x)**2 + (dy*cw_y)**2 <= alpha**2:
                k += 1
    return max(k, 1)

def laplace_noise_bound(eps, m, delta=1e-4):
    return (1.0/eps) * np.log(m / float(delta))

def cell_widths(xs, ys):
    return xs[1] - xs[0], ys[1] - ys[0]

# Fallback computation — EXACT SAME STYLE
if T is None:
    alpha = 0.002   # Crash DBSCAN eps in data units (~ lon/lat)
    cw_x, cw_y = cell_widths(xs, ys)
    kappa = estimate_kappa(alpha, cw_x, cw_y)
    noise_bound = laplace_noise_bound(eps_priv, num_grids)
    T = noise_bound / float(kappa)

print(f"[DP] Using T = {T:.4f}")

mask = noisy1 > T

# ================================================================
# 5. SPANS (exact call structure from moon_test.py)
# ================================================================
spans = ns.spans_from_noisy(noisy1, threshold_T=T)
print(f"[DP] Detected {len(spans)} spans")

# For visualization — same helper from moon_test.py
def spans_to_cell_labels(spans, shape):
    labels = np.full(shape, -1, dtype=int)
    for k, (x1,y1,x2,y2) in enumerate(spans):
        labels[x1:x2+1, y1:y2+1] = k
    return labels

cell_labels = spans_to_cell_labels(spans, H.shape)

# ================================================================
# 6. PLOTTING GRID (exact moon_test code)
# ================================================================
def plot_dp_grid_like_paper_from_labels(xs, ys, cell_labels, title="grid"):
    nx, ny = cell_labels.shape
    fig, ax = plt.subplots(figsize=(6,6))
    unique = np.unique(cell_labels)
    unique = unique[unique != -1]
    if len(unique) == 0:
        print("No spans.")
        return

    cmap = plt.cm.get_cmap("Spectral")(np.linspace(0,1,len(unique)))
    color_map = {lab: col for lab, col in zip(unique, cmap)}
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    for i in range(nx):
        for j in range(ny):
            lab = cell_labels[i,j]
            if lab == -1:
                continue
            x = xs[i]
            y = ys[j]
            rect = plt.Rectangle((x,y), dx, dy,
                                 edgecolor="black",
                                 facecolor=color_map[lab],
                                 fill=True)
            ax.add_patch(rect)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    plt.show()

plot_dp_grid_like_paper_from_labels(xs, ys, cell_labels,
                                    title=f"Crash DP grid (eps={eps_priv})")

# ================================================================
# 7. UTILITY EVAL (EXACT SAME BLOCK from moon_test.py)
# ================================================================
def boxes_to_world(spans, xs, ys):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    out = []
    for (x1,y1,x2,y2) in spans:
        out.append((xs[x1], ys[y1],
                    xs[x2] + dx, ys[y2] + dy))
    return out

def assign_labels_from_spans(points, spans_world):
    labels = np.full(len(points), -1, dtype=int)
    for k,(xmin,ymin,xmax,ymax) in enumerate(spans_world):
        m = ((points[:,0] >= xmin) & (points[:,0] < xmax) &
             (points[:,1] >= ymin) & (points[:,1] < ymax))
        labels[m] = k
    return labels

# "Truth" DBSCAN for Crash
y_true = DBSCAN(eps=0.001, min_samples=300, n_jobs=-1).fit_predict(X)
# --- after you compute y_true with DBSCAN ---
labels = y_true
unique = np.unique(labels)
n_clusters = (unique != -1).sum()
# (optional) subsample for faster plotting
plot_idx = np.arange(len(X))
if len(X) > 200_000:
    rng = np.random.default_rng(0)
    plot_idx = rng.choice(len(X), size=200_000, replace=False)
Xp = X[plot_idx]
lp = labels[plot_idx]
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
# noise points first (label == -1)
noise = (lp == -1)
plt.scatter(Xp[noise,0], Xp[noise,1], s=2, c="#cccccc", label="noise (-1)")
# clusters
mask = (lp != -1)
if mask.any():
    plt.scatter(Xp[mask,0], Xp[mask,1], s=2, c=lp[mask], cmap="tab20", label="clusters")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"DBSCAN clusters (eps=0.001, min_samples=300, k={n_clusters})")
plt.legend(loc="upper right", markerscale=6, frameon=True)
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

spans_world = boxes_to_world(spans, xs, ys)
y_pred = assign_labels_from_spans(X, spans_world)

covered = np.mean(y_pred != -1)
idx = (y_true != -1) & (y_pred != -1)

nmi = normalized_mutual_info_score(y_true[idx], y_pred[idx]) if idx.any() else 0
ari = adjusted_rand_score(y_true[idx], y_pred[idx]) if idx.any() else 0

print(f"[Utility] coverage={covered:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}")

# final scatter
plt.figure(figsize=(7,7))
noise_mask = (y_pred == -1)
plt.scatter(X[noise_mask,0], X[noise_mask,1], s=1, c="#d9d9d9")
plt.scatter(X[~noise_mask,0], X[~noise_mask,1], s=1, c=y_pred[~noise_mask], cmap="tab20")
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Crash DP-DBSCAN")
plt.show()