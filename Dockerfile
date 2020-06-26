#Download base image ubuntu 18.04
FROM ubuntu:18.04

LABEL maintainer="jannetta.steyn@newcastle.ac.uk"

ENV TZ=Europe/London

# Update Ubuntu Software repository
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get upgrade -y

# Install basic utilities and python
RUN apt-get -y install apt-utils
RUN apt-get -y install vim python3.6 python3.6-dev python3-pip wget 
RUN apt-get update
RUN apt-get upgrade -y

# Install Python 3 libraries
RUN pip3 install cython jupyter jupyter-core notebook sklearn pivottablejs GPyOpt pandas
RUN python3 -m pip install tensorflow

# Copy configuration file with password
RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter/.

# Create volume for persistency
VOLUME JupyterNotes
WORKDIR /JupyterNotes
EXPOSE 8080

RUN cd /JupyterNotes
# Run command to keep container running
CMD cd ; jupyter notebook --port=8080 --no-browser --ip=0.0.0.0 --allow-root
