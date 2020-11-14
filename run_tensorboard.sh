docker run -it -v $PWD/logs:/logs -p 6006-6015:6006-6015 tensorflow/tensorflow:latest-jupyter tensorboard --bind_all --logdir /logs
