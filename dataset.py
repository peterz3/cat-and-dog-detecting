import pathlib
import tensorflow as tf

data_root = 'dogs-vs-cats/train'

data_root = pathlib.Path(data_root)

dog_files = []
cat_files = []
for item in data_root.iterdir():
    if str(item)[19:22] == 'dog':
        dog_files.append(str(item))
    if str(item)[19:22] == 'cat':
        cat_files.append(str(item))


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

cat_labels = list(('cat') for name in  cat_files)
dog_labels = list(('dog') for name in  dog_files)




label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(cat_labels+dog_labels, tf.int64))

image_paths = tf.data.Dataset.from_tensor_slices(dog_files+ cat_files);


image_ds = image_paths.map(load_and_preprocess_image)




image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
num = len(cat_labels) + len(dog_labels)
ds = image_label_ds.shuffle(buffer_size=num)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)


