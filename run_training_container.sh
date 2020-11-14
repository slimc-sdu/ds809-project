

# https://www.tensorflow.org/install/docker
docker run --runtime=nvidia -it -v $PWD:/tmp -w /tmp/src tensorflow/tensorflow:latest-gpu bash

#"pip install -r requirements.txt && python ./train.py"