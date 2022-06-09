import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_image(image, r, c):
    plt.imshow(image.reshape((r, c)), cmap=cm.Greys_r)
    plt.show()
