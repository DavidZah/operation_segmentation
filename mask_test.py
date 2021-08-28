import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

y = np.zeros((1,) + (256,256) + (1,), dtype="uint8")
j = 0

img = load_img('Abyssinian_137.png', target_size=(256,256), color_mode="grayscale")

y[j] = np.expand_dims(img, 2)
# Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
y[j] -= 1
plt.imshow(y[j])
plt.show()
print('done')