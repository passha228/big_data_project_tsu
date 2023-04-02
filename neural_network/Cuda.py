import tensorflow as tf

if tf.test.is_gpu_available():
    print("TensorFlow использует GPU.")
else:
    print("TensorFlow не использует GPU.")