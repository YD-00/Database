import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
'''
This is the heart of the paper. Everything else (spans, clustering output) is post-processing.

What it does
Partition the 2D space into a grid.
For each cell, count points (sensitivity = 1).
Add Laplace noise with scale b = 1/epsilon_priv.
It's where the ε-DP guarantee lives.
Once this works, you can visualize heatmaps and already see the privacy-utility trade-off.

'''

import numpy as np
import noisy_grid_to_spans as ns

def plot_dp_grid_like_paper_from_labels(xs, ys, cell_labels,
                                        title="dp_dbscan_grid", figsize=(6,6)):
    """
    xs, ys: bin edges from make_grid (length nx+1, ny+1)
    cell_labels: (nx, ny) array with -1 for background, >=0 for cluster IDs
                 (x index = 0..nx-1, y index = 0..ny-1)
    """
    nx, ny = cell_labels.shape
    fig, ax = plt.subplots(figsize=figsize)

    # get unique cluster labels (ignore -1)
    unique = np.unique(cell_labels)
    unique = unique[unique != -1]
    if len(unique) == 0:
        print("[plot_dp_grid_like_paper_from_labels] No non-negative labels to plot.")
        return

    # color map per cluster (mimics Spectral colormap in the repo)
    cmap = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique)))
    color_map = {lab: col for lab, col in zip(unique, cmap)}
    color_map[-1] = np.array([0, 0, 0, 1])  # not actually used; we skip -1

    # plot limits: a little padded low/high
    x_low, x_high = xs[0], xs[-1]
    y_low, y_high = ys[0], ys[-1]
    x_off = 0.05 * (x_high - x_low)
    y_off = 0.05 * (y_high - y_low)
    ax.set_xlim(x_low - x_off, x_high + x_off)
    ax.set_ylim(y_low - y_off, y_high + y_off)

    # draw each labeled cell as a rectangle (like printer.plot_grid)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    for i in range(nx):       # x index
        for j in range(ny):   # y index
            lab = cell_labels[i, j]
            if lab == -1:
                continue
            color = color_map[lab]
            x = xs[i]
            y = ys[j]
            rect = plt.Rectangle(
                (x, y),
                dx,
                dy,
                edgecolor="black",
                facecolor=color,
                fill=True,
            )
            ax.add_patch(rect)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

def make_grid(points, xlim, ylim, n_x, n_y):
    # points: (N,2) array
    xs = np.linspace(xlim[0], xlim[1], n_x+1)
    ys = np.linspace(ylim[0], ylim[1], n_y+1)
    # digitize to bins
    xi = np.clip(np.digitize(points[:,0], xs) - 1, 0, n_x-1)
    yi = np.clip(np.digitize(points[:,1], ys) - 1, 0, n_y-1)
    H = np.zeros((n_x, n_y), dtype=float)
    np.add.at(H, (xi, yi), 1)  # histogram counts
    return H, xs, ys

def laplace_mech(counts, epsilon, clip_zero=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    b = 1.0 / float(epsilon)  # sensitivity=1
    noise = rng.laplace(loc=0.0, scale=b, size=counts.shape)
    out = counts + noise
    if clip_zero:
        out = np.maximum(out, 0.0)
    return out, noise

# Grid correctness test (no noise) ---
def test_grid_correctness():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, size=(1000, 2))
    H, xs, ys = make_grid(pts, (0,1), (0,1), nx=10, ny=10)
    # Expected: sum of histogram equals number of points
    assert abs(H.sum() - len(pts)) < 1e-9

# Noise distribution sanity ---
def test_noise_stats():
    rng = np.random.default_rng(1)
    counts = np.ones((64,64)) * 10  # arbitrary baseline
    eps = 1.0
    _, noise = laplace_mech(counts, eps, clip_zero=False, rng=rng)
    b = 1.0/eps
    # Laplace(0,b) has Var = 2*b^2, mean = 0
    mean = noise.mean()
    var = noise.var()
    assert abs(mean) < 0.1  # loose tolerance
    assert abs(var - 2*(b**2)) < 0.2  # loose tolerance


# --- 3) Epsilon scaling ---
def test_epsilon_scaling():
    rng = np.random.default_rng(2)
    counts = np.zeros((128,128))
    _, noise_small_eps = laplace_mech(counts, epsilon=0.5, clip_zero=False, rng=rng)
    rng = np.random.default_rng(2)
    _, noise_big_eps = laplace_mech(counts, epsilon=2.0, clip_zero=False, rng=rng)
    # std should shrink roughly in proportion to 1/epsilon
    assert noise_big_eps.std() < noise_small_eps.std() * 0.35  # 0.5/2.0 = 0.25 (allow slack)

# --- 4) Neighboring-dataset indistinguishability (empirical) ---
def sample_mechanism_outputs(counts, epsilon, trials=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Return one scalar summary (e.g., total mass above a threshold)
    vals = []
    for _ in range(trials):
        noisy, _ = laplace_mech(counts, epsilon, clip_zero=True, rng=rng)
        vals.append( (noisy >= 1.0).sum() )  # number of "dense" cells
    return np.array(vals)

def test_neighboring_similarity():
    rng = np.random.default_rng(3)
    base = np.zeros((32,32))
    base[10,10] = 9  # a small bump
    D = base.copy()
    Dprime = base.copy()
    # Neighboring datasets differ by ONE count (simulating one point)
    D[10,10] += 1  # +1 point in that cell

    eps = 1.0
    vD = sample_mechanism_outputs(D, eps, trials=1000, rng=np.random.default_rng(4))
    vDp = sample_mechanism_outputs(Dprime, eps, trials=1000, rng=np.random.default_rng(5))
    # The two empirical distributions should be close; KS statistic small
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(vD, vDp)
    assert p > 0.01  # fail only if trivially separable

# --- 5) Clipping bias smoke test ---
def test_clipping_effect_small_counts():
    rng = np.random.default_rng(6)
    counts = np.full((64,64), 0.5)  # small baseline
    eps = 1.0
    noisy_clip, _ = laplace_mech(counts, eps, clip_zero=True, rng=rng)
    noisy_raw, _  = laplace_mech(counts, eps, clip_zero=False, rng=np.random.default_rng(6))
    # Clipping raises mean a bit but shouldn't explode
    assert noisy_clip.mean() - noisy_raw.mean() < 0.2

def spans_to_cell_labels(spans, shape):
    """
    spans: list of (x1, y1, x2, y2) in *cell indices* (like your ns.spans_from_noisy)
    shape: (nx, ny) = H.shape
    Returns: labels[nx, ny], with -1 for non-core cells,
             0,1,2,... for different span components.
    """
    labels = np.full(shape, -1, dtype=int)
    for k, (x1, y1, x2, y2) in enumerate(spans):
        labels[x1:x2+1, y1:y2+1] = k
    return labels

# generate data
X, _ = make_moons(n_samples=2000, noise=0.08, random_state=0)

# grid + counts
H, xs, ys = make_grid(X, (X[:,0].min(), X[:,0].max()),
                         (X[:,1].min(), X[:,1].max()),
                         n_x=64, n_y=64)

# noise with different eps
noisy1, _ = laplace_mech(H, epsilon=4, clip_zero=True)        # eps: Privacy budget. Hiher-Less noise → clearer moons, less privacy； Lower→ More noise → blurrier, more privacy
noisy2, _ = laplace_mech(H, epsilon=0.5, clip_zero=True)

# Check DP histogram
# plot
fig, axs = plt.subplots(1,3, figsize=(12,3.5))
axs[0].imshow(H.T, origin='lower', aspect='auto'); axs[0].set_title('True counts')
axs[1].imshow(noisy1.T, origin='lower', aspect='auto'); axs[1].set_title('Noisy (ε=2.0)')
axs[2].imshow(noisy2.T, origin='lower', aspect='auto'); axs[2].set_title('Noisy (ε=0.5)')
for ax in axs: ax.axis('off')
plt.tight_layout(); plt.show()


# Check noisy span
# Choose a threshold: try 5,10,15
T = 2       # Density threshold for merging spans. higher-Merges more spans → smoother shapes but risk merging distinct clusters
spans = ns.spans_from_noisy(noisy1, threshold_T=T)
print(f"Detected {len(spans)} spans")

# Visualize
plt.imshow(noisy1.T, origin='lower', cmap='viridis')
for (x1,y1,x2,y2) in spans:
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], 'r-', lw=2)
plt.title(f"Eps=3, T={T}")
plt.show()

cell_labels = spans_to_cell_labels(spans, H.shape)   # shape = (nx, ny)
plot_dp_grid_like_paper_from_labels(xs, ys, cell_labels,
                                    title=f"dp_dbscan_eps{1}_grid")

### Utility check
# === Quick Utility Test ===
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

def boxes_to_world(spans, xs, ys):
    dx, dy = xs[1]-xs[0], ys[1]-ys[0]
    world = []
    for (x1,y1,x2,y2) in spans:
        world.append((xs[x1], ys[y1], xs[x2]+dx, ys[y2]+dy))
    return world

def assign_labels_from_spans(points, spans_world):
    labels = np.full(len(points), -1, dtype=int)
    for k,(xmin,ymin,xmax,ymax) in enumerate(spans_world):
        mask = (
            (points[:,0] >= xmin) & (points[:,0] < xmax) &
            (points[:,1] >= ymin) & (points[:,1] < ymax)
        )
        labels[mask] = k
    return labels

# 1) Truth via non-private DBSCAN on raw data (tune eps/min_samples to your moons scale)
y_true = DBSCAN(eps=0.2, min_samples=10).fit_predict(X)         # True label            HOW TO TUNE eps/min_samples to match DP-DBSCAN's?

# 2) Pred via spans
spans_world = boxes_to_world(spans, xs, ys)                     
y_pred = assign_labels_from_spans(X, spans_world)               # Noisy DP label

# 3) Quick metrics
covered = np.mean(y_pred != -1)  # coverage (% points assigned to some span)
# Only compare points that are non-noise in both views to avoid skew
# Coverage (~1) (% points of true clusters covered by any span):  
# NMI (0.6-0.8) (Normalized Mutual Information): Higher the better the cluster structure matches the ground truth, regardless of label IDs. 
# ARI (0.6-0.8) (Adjusted Rand Index): ARI compares how similar two clusterings are:
# 1. True labels: what the non-private (regular) DBSCAN found.
# 2. Predicted labels: what your differentially private spans produce. Higher the better.
idx = (y_true != -1) & (y_pred != -1)
nmi = normalized_mutual_info_score(y_true[idx], y_pred[idx]) if idx.any() else 0.0
ari = adjusted_rand_score(y_true[idx], y_pred[idx]) if idx.any() else 0.0

print(f"[Utility] coverage={covered:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}, labels={y_pred}, label_size={max(y_pred)}")
