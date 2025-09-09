import matplotlib.pyplot as plt
import numpy as np


def plot_simplest_camera_model(
        f=200,
        obj_x=1000,
        obj_h=200,
        objs_x_unk=[400],
):
    # Define the focal length

    plt.figure(figsize=(10, 4))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # object x
    plt.scatter([obj_x], [obj_h], color='g')
    plt.plot([obj_x, obj_x], [0, obj_h], color='g')
    plt.plot([0, obj_x], [0, 0], color='g')
    # projection x
    plt.plot([0, obj_x], [0, obj_h], color='g', linestyle='--', linewidth=0.7)

    plt.text(obj_x + 5, obj_h / 2, "w", fontsize=12, color='g')
    plt.text(obj_x / 2, 5, "d", fontsize=12, color='g')
    
    for x in objs_x_unk:
        # object x2
        plt.scatter([x], [obj_h], color='r')
        plt.plot([x, x], [0, obj_h], color='r')
        # projection x2
        plt.plot([0, x], [0, obj_h], color='r', linestyle='--', linewidth=0.7)

    plt.plot([0, f], [0, 0], color='b')
    plt.axvline(f, color='b')
    plt.text(f / 2, -20, "F = ?", color='b')
    plt.text(f + 5, 15, "p", color='b')

    plt.grid()
    plt.axis('equal')
    plt.xlabel("z (depth)")
    plt.ylabel("x (ancho)")

    plt.show()
