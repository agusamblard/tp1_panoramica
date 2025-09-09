import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_distortion(img_w, img_h, K, dist_coeffs, grid_step=20):
    # Generate a grid of points in the image
    x, y = np.meshgrid(np.arange(0, img_w, grid_step), np.arange(0, img_h, grid_step))
    points = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)

    # Undistort points using the provided camera matrix and distortion coefficients
    points_undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist_coeffs, P=K)
    points_undistorted = points_undistorted.reshape(-1, 2)

    # Calculate the distortion by finding the difference between the distorted and undistorted points
    distortion = points_undistorted - points
    distortion_magnitude = np.sqrt(distortion[:, 0] ** 2 + distortion[:, 1] ** 2)

    # Reshape for contour plotting
    distortion_magnitude_grid = distortion_magnitude.reshape(x.shape)

    # Plot the original points
    plt.figure(figsize=(10, 10))
    plt.imshow(np.zeros((img_h, img_w)), cmap='gray')
    # plt.scatter(points[:, 0], points[:, 1], color='red', s=5, label='Original points')

    # Use quiver to visualize the distortion as arrows
    plt.quiver(points[:, 0], points[:, 1], distortion[:, 0], distortion[:, 1], angles='xy', scale_units='xy', scale=1,
               color='white')

    # Add contour lines to represent the magnitude of distortion
    contour_levels = np.linspace(distortion_magnitude_grid.min(), distortion_magnitude_grid.max(), 10)
    plt.contour(x, y, distortion_magnitude_grid, levels=contour_levels, cmap='jet')

    plt.title("Distortion Visualization with Contours")
    plt.xlim([0, img_w])
    plt.ylim([img_h, 0])
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def visualize_3d_distortion(img_w, img_h, K, dist_coeffs, grid_step=20):
    # Generate a grid of points in the image
    x, y = np.meshgrid(np.arange(0, img_w, grid_step), np.arange(0, img_h, grid_step))
    points = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)

    # Undistort points using the provided camera matrix and distortion coefficients
    points_undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist_coeffs, P=K)
    points_undistorted = points_undistorted.reshape(-1, 2)

    # Calculate the distortion by finding the difference between the distorted and undistorted points
    distortion = points_undistorted - points
    distortion_magnitude = np.sqrt(distortion[:, 0] ** 2 + distortion[:, 1] ** 2)

    # Reshape for plotting
    x_grid = x.ravel()
    y_grid = y.ravel()
    z_grid = distortion_magnitude

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D surface
    ax.plot_trisurf(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none')

    # Add labels and title
    ax.set_title("3D Distortion Magnitude Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Distortion Magnitude")

    # Set axis limits
    ax.set_xlim(0, img_w)
    ax.set_ylim(0, img_h)
    ax.set_zlim(0, np.max(distortion_magnitude))

    # Show the plot
    plt.show()


def calculate_radial_distortion(K, dist_coeffs, points):
    """Calculates radial distortion effects on image points."""
    k1, k2, p1, p2, k3 = dist_coeffs

    # Normalize the points
    x = (points[:, 0] - K[0, 2]) / K[0, 0]
    y = (points[:, 1] - K[1, 2]) / K[1, 1]

    # Calculate r^2
    r2 = x ** 2 + y ** 2
    radial_distortion = (k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    #radial_distortion = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)

    # Apply radial distortion
    x_radial = x * radial_distortion
    y_radial = y * radial_distortion

    return x_radial, y_radial


def calculate_tangential_distortion(K, dist_coeffs, points):
    """Calculates tangential distortion effects on image points."""
    k1, k2, p1, p2, k3 = dist_coeffs

    # Normalize the points
    x = (points[:, 0] - K[0, 2]) / K[0, 0]
    y = (points[:, 1] - K[1, 2]) / K[1, 1]

    # Calculate tangential distortion
    r2 = x ** 2 + y ** 2
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    return x_tangential, y_tangential


def visualize_3d_distortions(img_w, img_h, K, dist_coeffs, grid_step=20, zlim=None):
    # Generate a grid of points in the image
    x, y = np.meshgrid(np.arange(0, img_w, grid_step), np.arange(0, img_h, grid_step))
    points = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)

    # Calculate radial and tangential distortions separately
    x_radial, y_radial = calculate_radial_distortion(K, dist_coeffs, points)
    x_tangential, y_tangential = calculate_tangential_distortion(K, dist_coeffs, points)

    radial_distortion_magnitude = np.sqrt(x_radial ** 2 + y_radial ** 2)
    tangential_distortion_magnitude = np.sqrt(x_tangential ** 2 + y_tangential ** 2)

    # Create 3D plots for radial and tangential distortions
    fig = plt.figure(figsize=(16, 8))

    # Radial distortion plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(x.ravel(), y.ravel(), radial_distortion_magnitude, cmap='viridis', edgecolor='none')
    ax1.set_title("3D Radial Distortion Magnitude Plot")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Radial Distortion Magnitude")
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(0, img_h)
    if zlim is not None:
        use_zlim = zlim
    else:
        use_zlim = np.max(radial_distortion_magnitude)
    ax1.set_zlim(0, use_zlim)

    # Tangential distortion plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(x.ravel(), y.ravel(), tangential_distortion_magnitude, cmap='plasma', edgecolor='none')
    ax2.set_title("3D Tangential Distortion Magnitude Plot")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Tangential Distortion Magnitude")
    ax2.set_xlim(0, img_w)
    ax2.set_ylim(0, img_h)
    if zlim is not None:
        use_zlim = zlim
    else:
        use_zlim = np.max(tangential_distortion_magnitude)
    ax2.set_zlim(0, use_zlim)


    plt.show()


def radial_distortion(K, dist_coeffs, points):
    """Calculates radial distortion effects on image points."""
    k1, k2, p1, p2, k3 = dist_coeffs

    # Normalize the points
    x = (points[:, 0] - K[0, 2]) / K[0, 0]
    y = (points[:, 1] - K[1, 2]) / K[1, 1]

    # Calculate r^2
    r2 = x ** 2 + y ** 2
    # radial_distortion = (k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    radial_distortion = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)

    # Apply radial distortion
    # x_distorted = x + x * radial_distortion
    # y_distorted = y + y * radial_distortion
    x_distorted = x * radial_distortion
    y_distorted = y * radial_distortion

    # Convert back to image coordinates
    x_distorted = x_distorted * K[0, 0] + K[0, 2]
    y_distorted = y_distorted * K[1, 1] + K[1, 2]

    return x_distorted, y_distorted


def tangential_distortion(K, dist_coeffs, points):
    """Calculates tangential distortion effects on image points."""
    k1, k2, p1, p2, k3 = dist_coeffs

    # Normalize the points
    x = (points[:, 0] - K[0, 2]) / K[0, 0]
    y = (points[:, 1] - K[1, 2]) / K[1, 1]

    # Calculate r^2
    r2 = x ** 2 + y ** 2

    # Calculate tangential distortion
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    # Apply tangential distortion
    x_distorted = x + x_tangential
    y_distorted = y + y_tangential

    # Convert back to image coordinates
    x_distorted = x_distorted * K[0, 0] + K[0, 2]
    y_distorted = y_distorted * K[1, 1] + K[1, 2]

    return x_distorted, y_distorted


def visualize_2d_distortions(img_w, img_h, K, dist_coeffs, grid_step=20):
    # Generate a grid of points in the image
    x, y = np.meshgrid(np.arange(0, img_w, grid_step), np.arange(0, img_h, grid_step))
    points = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)

    # Calculate radial and tangential distortions separately
    x_radial, y_radial = radial_distortion(K, dist_coeffs, points)
    x_tangential, y_tangential = tangential_distortion(K, dist_coeffs, points)

    # Calculate the distortion vectors by finding the difference between distorted and original points
    radial_distortion_vectors = np.stack((-x_radial + points[:, 0], -y_radial + points[:, 1]), axis=-1)
    tangential_distortion_vectors = np.stack((-x_tangential + points[:, 0], -y_tangential + points[:, 1]), axis=-1)

    # Create 2D quiver plot for radial distortion
    plt.figure(figsize=(8, 6))
    plt.quiver(points[:, 0], points[:, 1], radial_distortion_vectors[:, 0], radial_distortion_vectors[:, 1],
               color='white', angles='xy', scale_units='xy', scale=1)
    plt.scatter(K[0, 2], K[1, 2], s=10, c='y', marker='x')
    plt.title("2D Radial Distortion Vector Field")
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0)  # Invert y-axis for image-like coordinates
    plt.gca().set_facecolor('black')  # Set background color to black for better visibility
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio matches image size
    plt.show()

    # Create 2D quiver plot for tangential distortion
    plt.figure(figsize=(8, 6))
    plt.quiver(points[:, 0], points[:, 1], tangential_distortion_vectors[:, 0], tangential_distortion_vectors[:, 1],
               color='white', angles='xy', scale_units='xy', scale=1)
    plt.scatter(K[0, 2], K[1, 2], s=10, c='y', marker='x')
    plt.title("2D Tangential Distortion Vector Field")
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0)  # Invert y-axis for image-like coordinates
    plt.gca().set_facecolor('black')  # Set background color to black for better visibility
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio matches image size
    plt.show()


    # Calculate combined distortion vectors
    combined_distortion_vectors = radial_distortion_vectors + tangential_distortion_vectors

    # Create 2D quiver plot for combined distortion
    plt.figure(figsize=(8, 6))
    plt.quiver(points[:, 0], points[:, 1], combined_distortion_vectors[:, 0], combined_distortion_vectors[:, 1],
               color='white', angles='xy', scale_units='xy', scale=1)
    plt.scatter(K[0, 2], K[1, 2], s=10, c='y', marker='x')
    plt.title("2D Combined Radial + Tangential Distortion Vector Field")
    plt.xlim(0, img_w)
    plt.ylim(img_h, 0)  # Invert y-axis for image-like coordinates
    plt.gca().set_facecolor('black')  # Set background color to black for better visibility
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio matches image size
    plt.show()


if __name__ == '__main__':
    # Example usage
    img_w = 1920  # Image width
    img_h = 1080  # Image height
    K = np.array([
        [1456.12, 0., 907.729],
        [0., 1458.881, 543.169],
        [0., 0., 1.]
    ])

    focal_len = [
        K[0, 0], K[1, 1],
    ]  # Focal lengths
    cc = [
        K[0, 2],
        K[1, 2],
    ]  # Principal point
    dist_coeffs = np.array([
        0.06872788, -0.22046314, -0.00460212, -0.00284086, 0.18672138
    ])  # Distortion coefficients
    alpha_c = 0.0  # Skew coefficient

    # visualize_distortion(
    #     img_w,
    #     img_h,
    #     K,
    #     dist_coeffs,
    #     # alpha_c
    # )

    # visualize_3d_distortion(
    #     img_w,
    #     img_h,
    #     K,
    #     dist_coeffs,
    # )

    # visualize_3d_distortions(
    #     img_w, img_h, K, dist_coeffs
    # )

    visualize_2d_distortions(
        img_w, img_h, K, dist_coeffs
    )
