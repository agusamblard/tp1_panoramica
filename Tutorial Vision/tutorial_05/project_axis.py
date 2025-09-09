import itertools

import cv2
import numpy as np

from calib import board_points, detect_board

checkerboard = (10, 7)
square_size_mm = 24.2

object_points = square_size_mm * board_points(checkerboard)


K =  np.array([
        [   600.710,         0.000,        950.535],
        [     0.000,       602.324,        539.565],
        [     0.000,         0.000,          1.000]
])

# Distortion Coefficients :

dist_coeffs =  np.array([
        [  0.020128,     -0.033730,      -0.003931,       0.000204,       0.006070]
])

# K = np.array([
#     [606.628, 0.000, 949.399],
#     [0.000, 608.002, 535.278],
#     [0.000, 0.000, 1.000]
# ])
#
# # Distortion Coefficients :
#
# dist_coeffs = np.array([
#     [0.015396, -0.028823, -0.003059, -0.000634, 0.004652]
# ])

# K = np.array([
#     [954.070, 0.000, 647.905],
#     [0.000, 955.997, 365.002],
#     [0.000, 0.000, 1.000]
# ])
#
# # Distortion Coefficients :
#
# dist_coeffs = np.array([
#     [-0.016884, 0.277200, -0.003530, 0.000187, -0.652005]
# ])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def draw_line(img, pt1, pt2, color, thickness=3):
    pt1 = (np.round(pt1[0]).astype(int), np.round(pt1[1]).astype(int))
    pt2 = (np.round(pt2[0]).astype(int), np.round(pt2[1]).astype(int))
    ret = cv2.line(img, pt1, pt2, color, thickness)
    return ret


def plot_axis(image, rv, tv):

    axis_len_x = 9 * square_size_mm
    axis_len_y = 6 * square_size_mm
    axis_len_z = -6 * square_size_mm
    axis_points = np.array([[0, 0, 0],
                            [axis_len_x, 0, 0],
                            [0, axis_len_y, 0],
                            [0, 0, axis_len_z]], dtype=np.float32)

    # Project 3D points to the 2D image plane
    axis_points, _ = cv2.projectPoints(axis_points, rv, tv, K, dist_coeffs)

    # Draw the axes on the image
    axis_points = axis_points.reshape(-1, 2)
    origin = tuple(axis_points[0].ravel())

    draw_line(image, origin, axis_points[1], (0, 0, 255), 10)
    draw_line(image, origin, axis_points[2], (0, 255, 0), 10)
    draw_line(image, origin, axis_points[3], (255, 0, 0), 10)

    return image


def paint_squares(image, rv, tv, color=(255, 0, 255)):
    """
    More precise version that paints entire black squares purple.
    """
    width, height = checkerboard

    for i, j in itertools.product(
            range(width-1),
            range(height-1)
    ):

        # Only process black squares
        if (i + j) % 2 == 0:
            # Calculate 4 corners of the square in 3D
            x1 = i * square_size_mm
            y1 = j * square_size_mm
            x2 = x1 + square_size_mm
            y2 = y1 + square_size_mm

            square_corners = np.array([
                [x1, y1, 0],
                [x2, y1, 0],
                [x2, y2, 0],
                [x1, y2, 0]
            ], dtype=np.float32)

            # Project corners to 2D
            projected, _ = cv2.projectPoints(square_corners, rv, tv, K, dist_coeffs)
            projected = projected.reshape(-1, 2).astype(int)

            # Draw filled polygon
            cv2.fillPoly(image, [projected], color)

    return image


def plot_cube(image, rv, tv, position, cube_size=50, color=(0, 255, 255), thickness=2):
    """
    Draws a 3D cube at the specified world position in the image.

    Args:
        image: Input image (can be grayscale or color)
        rv: Rotation vector of the camera
        tv: Translation vector of the camera
        position: 3D world coordinates (x,y,z) where cube will be centered
        cube_size: Size of the cube in mm
        color: BGR color tuple for the cube edges
        thickness: Line thickness in pixels
    Returns:
        Image with cube drawn
    """
    half_size = cube_size / 2

    # Define the 8 corners of the cube relative to center position
    cube_points = np.array([
        # Bottom face (rear-left is origin)
        [0, 0, 0],  # 0: bottom-rear-left (anchor point)
        [cube_size, 0, 0],  # 1: bottom-rear-right
        [cube_size, cube_size, 0],  # 2: bottom-front-right
        [0, cube_size, 0],  # 3: bottom-front-left

        # Top face
        [0, 0, -cube_size],  # 4: top-rear-left
        [cube_size, 0, -cube_size],  # 5: top-rear-right
        [cube_size, cube_size, -cube_size],  # 6: top-front-right
        [0, cube_size, -cube_size]  # 7: top-front-left
    ], dtype=np.float32)

    # Offset by the desired position
    cube_points += np.array(position, dtype=np.float32)

    # Project 3D points to 2D image plane
    projected, _ = cv2.projectPoints(cube_points, rv, tv, K, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(int)

    # Define the 12 edges of the cube
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Draw all edges
    for start, end in edges:
        start_pt = tuple(projected[start])
        end_pt = tuple(projected[end])
        cv2.line(image, start_pt, end_pt, color, thickness)



def run():
    source = 0
    cap = cv2.VideoCapture(source)
    # 3840x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    #cv2.namedWindow("object", cv2.WINDOW_NORMAL)
    #cv2.imshow("object", method.object_image)

    while True:
        ret, frame = cap.read()
        # print(frame.shape)
        frame = frame[:, 0:1920, :]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vis = frame.copy()

        found, corners = detect_board(checkerboard, gray, scale=0.5)
        if corners is not None:

            try:
                pose = cv2.solvePnP(
                    object_points,
                    corners,
                    K,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )
                ok, rv, tv = pose

                paint_squares(vis, rv, tv)
                # plot_axis(vis, rv, tv)
                plot_cube(vis, rv, tv,
                          position=(2 * square_size_mm,
                                    2 * square_size_mm,
                                    0 * -square_size_mm),
                          cube_size=3 * square_size_mm
                )

            except Exception as e:
                pass

        cv2.imshow("frame", vis)

        key = cv2.waitKey(5)  # & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
