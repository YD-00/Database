from collections import deque
import numpy as np
from scipy.ndimage import label

def connected_components(mask):
    # mask: boolean 2D array for dense cells
    n_x, n_y = mask.shape
    spans = []
    seen = np.zeros_like(mask, dtype=bool)
    for i in range(n_x):
        for j in range(n_y):
            if mask[i,j] and not seen[i,j]:
                comp = []
                q = deque([(i,j)])
                seen[i,j] = True
                while q:
                    x,y = q.popleft()
                    comp.append((x,y))
                    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:  # 4-neigh
                        u,v = x+dx, y+dy
                        if 0<=u<n_x and 0<=v<n_y and mask[u,v] and not seen[u,v]:
                            seen[u,v] = True
                            q.append((u,v))
                spans.append(comp)
    return spans

def spans_from_noisy(noisy_counts, threshold_T):
    dense = noisy_counts >= threshold_T
    comps = connected_components(dense)
    # represent each span as a min/max box over cell indices
    boxes = []
    for comp in comps:
        xs = [c[0] for c in comp]; ys = [c[1] for c in comp]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def spans_from_noisy_with_neighbors(
    mask,
    noisy_counts,
    alpha,             # = DBSCAN eps in data units
    cell_wx, cell_wy,  # grid cell size in x/y
    T,                 # threshold computed via repo rule
    min_cells=5        # prune tiny components
):
    # 1) neighbor sum over a disk of radius alpha
    S = neighbor_sum_disk(noisy_counts, alpha, cell_wx, cell_wy)  # uses your function
    # 2) threshold on neighbor-sum (not raw noisy counts!)
    # core_mask = (S >= T)
    core_mask = mask
    # 3) 8-connected components → bounding boxes
    spans = merge_cells_to_spans(core_mask)  # your 8-neigh version
    # 4) prune tiny spans (optional but helpful)
    pruned = []
    for (x1,y1,x2,y2) in spans:
        area_cells = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area_cells >= min_cells:
            pruned.append((x1,y1,x2,y2))
    return pruned, S, core_mask


def neighbor_sum_disk(noisy, alpha, cell_wx, cell_wy):
    # build a circular structuring element in cell units
    rx = int(np.ceil(alpha / cell_wx))
    ry = int(np.ceil(alpha / cell_wy))
    rr = max(rx, ry)
    yy, xx = np.ogrid[-rr:rr+1, -rr:rr+1]
    disk = (xx*cell_wx)**2 + (yy*cell_wy)**2 <= alpha**2
    # convolve by binary mask (as sum over neighborhood)
    from scipy.signal import convolve2d
    return convolve2d(noisy, disk.astype(float), mode="same", boundary="fill", fillvalue=0.0)

def merge_cells_to_spans(core_mask):
    # 8-connectivity to avoid gaps
    structure = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=int)
    labeled, num = label(core_mask, structure=structure)
    spans = []
    for lab in range(1, num+1):
        ys, xs = np.where(labeled == lab)
        spans.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return spans


# Simple binary mask
mask = np.array([
    [0,1,1,0],
    [0,1,0,0],
    [0,0,1,1],
    [0,0,1,1]
], dtype=bool)

comps = connected_components(mask)
print(comps)