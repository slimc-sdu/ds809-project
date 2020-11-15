docker run -it --runtime=nvidia --net=host -v $PWD:/tf/notebooks/dm809-project -p 8888:8888 -p  6006-6015:6006-6015 tensorflow/tensorflow:latest-gpu-jupyter
