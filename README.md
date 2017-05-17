# tensorflow_demo

This is a demo about how to use TensorFlow with Docker to re-train the last loyer of a model, to predict the object in the images.
All you need is to install Docker on your local machine, you can [follow the steps here](https://docs.docker.com/engine/installation/).

Then follow the steps as following:

1. Run TensorFlow image in Docker:
'''
$ docker run -it gcr.io/tensorflow/tensorflow:latest-devel
'''
2. Download and un-zip the training set:
'''
$ cd ~

$ curl -O http://download.tensorflow.org/example_images/flower_photos.tgz

$ tar xzf flower_photos.tgz
'''
3. Download the python code of retraining the model:
'''
$ curl -O https://raw.githubusercontent.com/sophiesongge/tensorflow_demo/master/image_retrain.py
'''
4. Re-train the model:
'''
$ python image_retrain.py --image_dir flower_photos
'''
5. Use the re-trained model to predict an image:



