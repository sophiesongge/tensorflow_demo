import numpy as np
import tensorflow as tf
import cv2

import getopt
import sys

imagePath = ''
modelFullPath = '/tmp/output_graph.pb'
labelsFullPath = '/tmp/output_labels.txt'
outputPath = ''

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    orig = cv2.imread(imagePath)

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'r')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]].replace("\n", " ")
        cv2.putText(orig, "Output label: {}".format(answer), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (102, 0, 255), 2)
        prob = predictions[top_k[0]]
        cv2.putText(orig, "Prob: {0:.0f}%".format(prob * 100), (10, 90), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imwrite(outputPath, orig)
        return answer


if __name__ == '__main__':
    opts,args = getopt.getopt(sys.argv[1:],'-i:-m:-l:-o:',['imagePath=','modelFullPath=','labelsFullPath=','outputPath='])
    for opt_name,opt_value in opts:
        if opt_name in ('-i','--imagePath'):
            imagePath = opt_value
        if opt_name in ('-m','--modelFullPath'):
            modelFullPath = opt_value
        if opt_name in ('-l','--labelsFullPath'):
            labelsFullPath = opt_value
        if opt_name in ('-o','--outputPath'):
            outputPath = opt_value
    run_inference_on_image()

