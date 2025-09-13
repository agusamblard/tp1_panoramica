from Utils import dlt
import numpy as np
import cv2

def reprojection_errors(H, pts_src, pts_dst):
    """
    Error de reproyección unidireccional (a la imagen destino), en píxeles.
    e_i = || dst_i - norm(H * src_i) ||_2
    """
    n = len(pts_src)
    src_h = np.c_[pts_src, np.ones(n)]
    proj  = (H @ src_h.T).T
    proj  = proj[:, :2] / proj[:, [2]]
    return np.linalg.norm(pts_dst - proj, axis=1)

def symmetric_transfer_error(H, pts_src, pts_dst):
    """
    Error simétrico: e = ||x' - Hx|| + ||x - H^{-1}x'||.
    Un poco más caro pero más estable.
    """
    e1 = reprojection_errors(H, pts_src, pts_dst)
    try:
        Hinv = np.linalg.inv(H)
        e2 = reprojection_errors(Hinv, pts_dst, pts_src)
    except np.linalg.LinAlgError:
        e2 = np.full_like(e1, np.inf)
    return e1 + e2

def _non_degenerate(sample_src, sample_dst, tol=1e-3, min_area=10.0):
    """
    Evita muestras degeneradas:
    - chequea rango >= 3 (no colineales)
    - chequea que el área del cuadrilátero formado sea mayor a min_area
    """
    def rank3(pts):
        M = np.c_[pts, np.ones(len(pts))]
        return np.linalg.matrix_rank(M, tol)

    def area_quad(pts):
        # área del polígono con 4 puntos
        x = pts[:,0]; y = pts[:,1]
        return 0.5 * abs(
            x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]
            - y[0]*x[1] - y[1]*x[2] - y[2]*x[3] - y[3]*x[0]
        )

    return (rank3(sample_src) >= 3 and
            rank3(sample_dst) >= 3 and
            area_quad(sample_src) > min_area and
            area_quad(sample_dst) > min_area)


def ransac_homography(pts_src, pts_dst, thresh=3.0, max_trials=5000,
                      confidence=0.999, use_symmetric_error=True, seed=None):
    """
    Implementación RANSAC para homografía.
    Devuelve:
      - H_best (3x3)
      - inlier_mask (bool de tamaño N)
    """
    assert pts_src.shape == pts_dst.shape and pts_src.shape[0] >= 4
    N = len(pts_src)
    rng = np.random.default_rng(seed)

    best_inliers = None
    best_num = 0
    best_err_sum = np.inf

    s = 4  # número mínimo de puntos
    log1p = np.log

    trials = max_trials
    i = 0
    while i < trials:
        i += 1
        idx = rng.choice(N, size=s, replace=False)
        sample_src = pts_src[idx]
        sample_dst = pts_dst[idx]

        if not _non_degenerate(sample_src, sample_dst):
            continue

        try:
            H = dlt(sample_src, sample_dst)
        except np.linalg.LinAlgError:
            continue

        # calcular errores
        if use_symmetric_error:
            errs = symmetric_transfer_error(H, pts_src, pts_dst)
        else:
            errs = reprojection_errors(H, pts_src, pts_dst)

        inliers = errs < thresh
        num_inl = int(inliers.sum())

        if num_inl > best_num or (num_inl == best_num and errs[inliers].sum() < best_err_sum):
            best_num = num_inl
            best_inliers = inliers
            best_err_sum = errs[inliers].sum()

            # early stopping adaptativo
            w = max(1e-6, num_inl / N)
            num = 1 - confidence
            den = 1 - w ** s
            if den <= 1e-12:
                trials = i
            else:
                trials = min(trials, int(np.ceil(log1p(num) / log1p(den))))
                trials = max(trials, i + 1)

    if best_inliers is None or best_num < 4:
        raise RuntimeError("RANSAC no encontró un modelo válido.")

    # recalcular homografía con TODOS los inliers
    H_final = dlt(pts_src[best_inliers], pts_dst[best_inliers])
    H_final /= H_final[2,2]

    return H_final, best_inliers

def homography_final_opencv(pts_src, pts_dst, inlier_mask):
    """
    Homografía final usando SOLO los inliers, con cv2.findHomography sin RANSAC.
    (Esto es lo permitido por el enunciado.)
    """
    H, _ = cv2.findHomography(pts_src[inlier_mask], pts_dst[inlier_mask], method=0)
    return H / H[2, 2]