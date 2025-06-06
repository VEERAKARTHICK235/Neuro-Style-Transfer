import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os

# Ensure eager execution
tf.compat.v1.enable_eager_execution()

# Function to load and resize image
def load_img(path_to_img, max_dim=256):  # smaller = faster
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Convert tensor to PIL image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Paths
content_path = 'images/content.jpg'
style_path = 'images/style.jpg'

# Load images
content_image = load_img(content_path)
style_image = load_img(style_path)

# ✅ Load optimized pre-trained style transfer model from TF Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Stylize!
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Save and show result
result_image = tensor_to_image(stylized_image)
os.makedirs('output', exist_ok=True)
result_image.save('output/stylized_image.jpg')
print("✅ Style transfer complete! Output saved in output/stylized_image.jpg")

plt.imshow(result_image)
plt.axis('off')
plt.show()
