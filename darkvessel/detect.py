"""Cell-averaging CFAR ship detector with component grouping."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    x_m: float
    y_m: float
    peak: float          # peak / local clutter mean
    peak_raw: float      # peak intensity in units of the scene clutter mean
    area_px: int
    length_m: float


def _box_mean(img, k):
    return ndimage.uniform_filter(img, size=k, mode="reflect")


def estimate_texture_shape(work, eff_looks, trim=0.999):
    """Method-of-moments estimate of the K-distribution shape nu.

    For K clutter E[I^2]/E[I]^2 = (1 + 1/L)(1 + 1/nu). The brightest 0.1% of
    pixels are dropped so ships do not inflate the estimate.
    """
    v = work.ravel()
    v = v[v <= np.quantile(v, trim)]
    r = (v * v).mean() / v.mean() ** 2
    inv_nu = r / (1 + 1 / eff_looks) - 1
    return float(np.clip(1 / max(inv_nu, 1e-3), 0.5, 1e3))


def k_threshold(nu, eff_looks, pfa):
    """Threshold t (in units of the clutter mean) with P(I > t) = pfa under K."""
    from scipy.stats import gamma as G
    from scipy.optimize import brentq

    tau = np.linspace(1e-4, 1, 2000) ** 2 * (G.ppf(1 - 1e-9, nu, scale=1 / nu))
    w = G.pdf(tau, nu, scale=1 / nu)
    w /= np.trapezoid(w, tau)

    def tail(t):
        return np.trapezoid(w * G.sf(t / tau, eff_looks, scale=1 / eff_looks), tau)

    return brentq(lambda t: np.log(max(tail(t), 1e-300)) - np.log(pfa), 1.0, 500.0)


def cfar(img, pixel_m, guard_px=31, train_px=71, k=None, pfa=1e-6, looks=3,
         min_area_px=2, presum=2, model="k", group_px=4):
    """CA-CFAR on a presummed intensity image.

    model="k": threshold from a K-distribution whose shape is estimated from
    the scene itself, so spiky rough-sea clutter gets a higher threshold.
    model="gamma": classic speckle-only threshold (kept for comparison; it
    floods with false alarms once the sea has texture).
    """
    from scipy.stats import gamma as gamma_dist

    work = ndimage.uniform_filter(img, size=presum) if presum > 1 else img
    eff_looks = looks * presum * presum
    if k is None:
        if model == "k":
            k = k_threshold(estimate_texture_shape(work, eff_looks), eff_looks, pfa)
        else:
            k = gamma_dist.ppf(1 - pfa, eff_looks, scale=1.0 / eff_looks)

    n_out, n_in = train_px ** 2, guard_px ** 2
    mean_out = _box_mean(work, train_px) * n_out
    mean_in = _box_mean(work, guard_px) * n_in
    clutter = (mean_out - mean_in) / (n_out - n_in)
    clutter = np.maximum(clutter, 1e-6)

    mask = work > k * clutter
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    # bright scatterers on one hull often come out as separate blobs; group
    # blobs that lie within a few pixels, but measure only the hit pixels
    grouped = ndimage.binary_dilation(mask, iterations=group_px) if group_px else mask
    labels, n = ndimage.label(grouped, structure=np.ones((3, 3)))
    labels = labels * mask
    if n == 0:
        return [], mask

    scene_mean = float(np.median(clutter))
    dets = []
    objs = ndimage.find_objects(labels)
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = labels[sl] == lab
        area = int(sub.sum())
        if area < min_area_px:
            continue
        w = (work[sl] * sub)
        yy, xx = np.nonzero(sub)
        wts = w[yy, xx]
        cy = (yy * wts).sum() / wts.sum() + sl[0].start
        cx = (xx * wts).sum() / wts.sum() + sl[1].start
        if len(yy) > 1:
            cov = np.cov(np.vstack([xx, yy]), aweights=wts)
            length_px = 4.0 * np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0))
        else:
            length_px = 1.0
        peak = float((work[sl][sub] / clutter[sl][sub]).max())
        peak_raw = float(work[sl][sub].max() / scene_mean)
        dets.append(Detection(float(cx * pixel_m), float(cy * pixel_m), peak, peak_raw, area,
                              float(max(length_px, 1.0) * pixel_m)))
    return dets, mask
