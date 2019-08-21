import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    # Load .ckpt file
    saver = tf.train.import_meta_graph('./model/.meta')
    saver.restore(sess, './model/')

    # Save as .pb file
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['y_prediction/Add']
    )
    with tf.gfile.GFile('./model/model.pb', 'wb') as fid:
        serialized_graph = output_graph_def.SerializeToString()
        fid.write(serialized_graph)
