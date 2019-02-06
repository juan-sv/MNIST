FROM tensorflow/tensorflow:latest-py3

# print OS version
#RUN cat /etc/os-release
#RUN lsb_release -a

# Install git
RUN apt-get update
RUN apt-get -y install git-core

# Clone git repository
RUN git clone https://github.com/juan-sv/katib.git

# Set working directory
WORKDIR ./katib/