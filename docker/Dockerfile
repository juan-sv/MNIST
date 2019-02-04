FROM tensorflow/tensorflow:latest-py3

#RUN git clone https://github.com/apache/incubator-mxnet.git
#RUN ln -s /incubator-mxnet/example/image-classification/common common
#RUN ln -s /incubator-mxnet/example/image-classification/symbols symbols

ADD MNIST/MNIST.py /
ADD MNIST/run.py /
ADD requirements.txt /
#ADD common /common

#
#RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
#RUN python3 get-pip.py --user
#RUN pip install -r ../requirements.txt

#CMD [ "cd", "./MNIST" ]
#CMD [ "python", "../optimize_MNIST.py" ]
WORKDIR ./