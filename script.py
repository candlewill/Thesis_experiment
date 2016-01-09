from visualize import draw_iter
import numpy as np

train_loss = np.array([1.1743, 0.9886, 0.8962, 0.8822, 0.8274])
valid_loss = np.array([0.7978, 0.7697, 0.7139, 0.7076, 0.7158])
X = range(len(train_loss))
# , x_labels, y_labels, title

draw_iter(train_loss, valid_loss, x_labels='Epoch', y_labels='Loss')
