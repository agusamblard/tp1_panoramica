import matplotlib.pyplot as plt
import cv2

# Cargar la imagen
img = cv2.imread("img/edi0.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Lista para guardar los puntos
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        print(f"Punto seleccionado: ({x:.2f}, {y:.2f})")
        # Dibujar un punto en la imagen
        plt.scatter(x, y, c='r', s=50)
        plt.draw()

# Crear figura y conectar el evento del mouse
fig, ax = plt.subplots()
ax.imshow(img_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.title("Click en 4 puntos de interés")
plt.show()

# Después de cerrar la ventana, los puntos estarán en la lista
print("Puntos seleccionados:", points)