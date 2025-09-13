import cv2
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np


def detectar_descriptores(imagen, metodo="ORB", nfeatures=500):
    """
    Detecta keypoints y calcula descriptores de una imagen.
    
    Args:
        imagen (np.array): imagen cargada con cv2.imread
        metodo (str): "SIFT" o "ORB"
        nfeatures (int): cantidad máxima de features
    
    Returns:
        keypoints, descriptors
    """
    if metodo.upper() == "SIFT":
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    elif metodo.upper() == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    else:
        raise ValueError("Método no soportado. Usar 'SIFT' u 'ORB'.")

    keypoints, descriptors = detector.detectAndCompute(imagen, None)
    return keypoints, descriptors

def mostrar_keypoints(imagen, keypoints, titulo="Keypoints"):
    img_kp = imagen.copy()
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv2.circle(img_kp, (x, y), 5, (0,0,255), -1) 
    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def aplicar_anms(keypoints, N=500):
    """
    ANMS rápido usando KD-Tree.
    
    Args:
        keypoints: lista de cv2.KeyPoint detectados
        N: cantidad máxima de puntos a conservar
    
    Returns:
        keypoints_filtrados: lista reducida de keypoints
    """
    if len(keypoints) <= N:
        return keypoints

    # Extraer coordenadas y respuestas
    coords = np.array([kp.pt for kp in keypoints])
    respuestas = np.array([kp.response for kp in keypoints])

    # Ordena
    idx_sorted = np.argsort(-respuestas)
    coords_sorted = coords[idx_sorted]
    respuestas_sorted = respuestas[idx_sorted]

    # KDTree para búsquedas rápidas
    tree = cKDTree(coords_sorted)

    # Radios inicializados en infinito
    radios = np.full(len(coords_sorted), np.inf)

    # Para cada punto (menos fuerte), buscar el vecino más cercano más fuerte
    for i in range(1, len(coords_sorted)):
        d, j = tree.query(coords_sorted[i], k=i)  
        radios[i] = np.min(np.sum((coords_sorted[i] - coords_sorted[:i])**2, axis=1))

    # Ordenar por radio
    idx_final = np.argsort(-radios)

    # Seleccionar top N
    seleccionados_idx = idx_final[:N]
    seleccionados = [keypoints[idx_sorted[i]] for i in seleccionados_idx]

    return seleccionados

def filtrar_descriptores(keypoints, descriptores, keypoints_filtrados):
    """
    Retorna exactamente los descriptores correspondientes a los keypoints filtrados.
    Compara por coordenadas (redondeadas a int) en lugar de objetos KeyPoint.
    """
    coords_filtrados = {tuple(map(int, kp.pt)) for kp in keypoints_filtrados}
    kp_new, des_new = [], []
    for kp, des in zip(keypoints, descriptores):
        if tuple(map(int, kp.pt)) in coords_filtrados:
            kp_new.append(kp)
            des_new.append(des)
    return kp_new, np.array(des_new)


def match_features(desc1, desc2, metodo="SIFT", ratio=0.75,
                   use_ratio=True, use_cross_check=True):
    """
    Matching entre dos conjuntos de descriptores aplicando:
      - Lowe's ratio test
      - Cross-check
    Permite activar uno, otro, o ambos.
    Maneja el caso de que knnMatch devuelva menos de 2 matches.
    """
    if metodo.upper() == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN Matching (hasta 2 vecinos)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio (si corresponde)
    buenos = []
    for m_n in matches:
        if len(m_n) < 2:
            continue  
        m, n = m_n
        if not use_ratio or m.distance < ratio * n.distance:
            buenos.append(m)

    # Cross-check (si corresponde)
    if use_cross_check:
        matches_12 = {(m.queryIdx, m.trainIdx) for m in buenos}
        matches_back = bf.knnMatch(desc2, desc1, k=2)

        matches_21 = set()
        for m_n in matches_back:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if not use_ratio or m.distance < ratio * n.distance:
                matches_21.add((m.trainIdx, m.queryIdx))

        interseccion = matches_12 & matches_21
        buenos = [m for m in buenos if (m.queryIdx, m.trainIdx) in interseccion]

    return buenos


def dibujar_matches(img1, kp1, img2, kp2, matches, max_matches, titulo="Matches"):
    """
    Visualiza las correspondencias.
    """
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def dlt(ori, dst):
    """
    Estima homografía H con DLT (mínimos cuadrados).
    Admite >= 4 pares de puntos.
    """
    if ori.shape[0] < 4:
        raise ValueError("Se necesitan al menos 4 pares de puntos")

    A = []
    for (x, y), (xp, yp) in zip(ori, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)

    # SVD (más estable)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3,3)

    # Normalizar para que H[2,2] = 1
    return H / H[2,2]

def show_points(img_bgr, pts, title=""):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.scatter(pts[:,0], pts[:,1], s=70, marker='o',
                facecolors='none', edgecolors='yellow', linewidths=2)
    for i, (x, y) in enumerate(pts):
        plt.text(x+5, y-5, f"{i+1}", fontsize=12, color='black',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    plt.title(title); plt.axis('off'); plt.show()


def extraer_puntos(matches, kp1, kp2):
    """Convierte matches en arrays de puntos (src, dst)."""
    pts_src = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts_src, pts_dst

def dibujar_inliers(img1, kp1, img2, kp2, matches, inlier_mask, titulo="Matches con RANSAC"):
    """
    Dibuja inliers (verde) y outliers (rojo).
    """
    inliers  = [m for i, m in enumerate(matches) if inlier_mask[i]]
    outliers = [m for i, m in enumerate(matches) if not inlier_mask[i]]

    print(f"{titulo}: {len(inliers)} inliers, {len(outliers)} outliers (total {len(matches)})")

    img_inliers  = cv2.drawMatches(img1, kp1, img2, kp2, inliers, None,
                                matchColor=(0,255,0),  # verde fuerte
                                singlePointColor=None,
                                matchesThickness=3,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_outliers = cv2.drawMatches(img1, kp1, img2, kp2, outliers, None,
                                matchColor=(255,0,0),  # rojo fuerte
                                singlePointColor=None,
                                matchesThickness=3,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Fusionar ambas imágenes en un solo canvas
    overlay = cv2.addWeighted(img_inliers, 0.7, img_outliers, 0.7, 0)

    plt.figure(figsize=(15,8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def calcular_canvas_optimo(img_anchor, img_left, img_right, H_left, H_right):
    h, w = img_anchor.shape[:2]

    # Esquinas
    corners_anchor = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    corners_left   = np.array([[0,0],[img_left.shape[1],0],[img_left.shape[1],img_left.shape[0]],[0,img_left.shape[0]]], dtype=np.float32).reshape(-1,1,2)
    corners_right  = np.array([[0,0],[img_right.shape[1],0],[img_right.shape[1],img_right.shape[0]],[0,img_right.shape[0]]], dtype=np.float32).reshape(-1,1,2)

    # Proyección
    warped_left  = cv2.perspectiveTransform(corners_left,  H_left)
    warped_right = cv2.perspectiveTransform(corners_right, H_right)

    # Unir todas las esquinas
    all_corners = np.vstack((corners_anchor, warped_left, warped_right))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    # Traslación
    tx, ty = -xmin, -ymin
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)

    size = (xmax-xmin, ymax-ymin)

    return T, size

def blend_images_por_canal(warps, mostrar=True):
    """
    Aplica blending canal por canal, usando distanceTransform como máscara.
    """
    h, w = warps[0].shape[:2]
    acc_r = np.zeros((h, w), np.float32)
    acc_g = np.zeros((h, w), np.float32)
    acc_b = np.zeros((h, w), np.float32)
    weights_sum = np.zeros((h, w), np.float32)

    mascaras = []

    for warp in warps:
        # máscara binaria de la imagen warp
        mask = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            continue

        # máscara degradada
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        mascaras.append(dist)

        # separar canales
        r, g, b = cv2.split(warp.astype(np.float32))

        # aplicar máscara canal por canal
        acc_r += r * dist
        acc_g += g * dist
        acc_b += b * dist

        weights_sum += dist

    # evitar división por cero
    weights_sum[weights_sum == 0] = 1.0

    # normalizar cada canal
    r_final = acc_r / weights_sum
    g_final = acc_g / weights_sum
    b_final = acc_b / weights_sum

    # unir en una sola imagen
    panorama = cv2.merge([r_final, g_final, b_final]).astype(np.uint8)

    if mostrar:
        fig, axes = plt.subplots(1, len(mascaras), figsize=(15,5))
        for i, W in enumerate(mascaras):
            axes[i].imshow(W, cmap="gray")
            axes[i].set_title(f"Máscara canal {i}")
            axes[i].axis("off")
        plt.suptitle("Máscaras de blending aplicadas por canal")
        plt.show()

        plt.figure(figsize=(15,8))
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title("Panorámica con blending por canal (3.7)")
        plt.axis("off")
        plt.show()

    return panorama
