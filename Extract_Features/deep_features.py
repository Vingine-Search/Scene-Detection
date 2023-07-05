import tensorflow as tf
import cv2
import os
from .utils import *
def DeepFeaturesExtractor():
    inputs = tf.keras.layers.Input((299, 299, 3))
    xcp_preprocessed_inputs = tf.keras.applications.xception.preprocess_input(inputs)
    xcp = tf.keras.applications.Xception(
        input_tensor=xcp_preprocessed_inputs,
        # Don't include the top classification layer, as we will use this model for feature extraction.
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    # For sanity, we shouldn't run `model.fit` anyway.
    xcp.trainable = False
    features = tf.keras.layers.Flatten()(xcp.output)
    return tf.keras.Model(inputs, features)
def get_deep_features_for_key_frames(keyframe_files: list):
    """Returns feature vectors based on `Xception` CNN network for the given list of (key) frames.
    Args:
        keyframe_files: A list of string file names for the frames to extract deep features from.
                        (e.g. key_frames_files=['shot1_keyframe.jpg', 'shot2_keyframe.jpg', ...])
    """
    xcp_feat_ext = DeepFeaturesExtractor()
    image_dimensions = xcp_feat_ext.input.shape[1:]
    # A generator to load the key frames asynchronously.
    def image_generator():
        for keyframe_file in keyframe_files:
            image = tf.image.decode_image(tf.io.read_file(keyframe_file))
            resized_image = tf.image.resize_with_pad(
                image, target_height=image_dimensions[0], target_width=image_dimensions[1])
            yield resized_image
    # Batch every one example in a dataset.
    dataset = tf.data.Dataset.from_generator(
        image_generator, output_shapes=image_dimensions, output_types=(tf.float32)).batch(1)
    return xcp_feat_ext.predict(dataset)

if __name__ == "__main__":
    output_dir_shot_boundries = "../Dataset/shot_boundary"
    images = []

    for filename in os.listdir(output_dir_shot_boundries):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(output_dir_shot_boundries, filename))
            images.append(image)
    ret = get_deep_features_for_key_frames(images)
    distance_matrix = cosine_dist(ret)
    D_sum = get_optimal_sequence_add2(distance_matrix,len(ret))