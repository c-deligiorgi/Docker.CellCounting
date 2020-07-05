#Download base image python:3.6-stretch
FROM python:3.6-stretch

LABEL maintainer="C.Deligiorgi@newcastle.ac.uk"

ENV TZ=Europe/London

# Install Python 3 libraries
RUN pip install cython jupyter jupyter-core notebook sklearn pivottablejs GPyOpt pandas
RUN pip install tensorflow
RUN pip install numpy scipy
RUN pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.3.1 torchvision==0.4.2
RUN pip install tqdm

# Copy configuration file with password
RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter/.

# Define the working directory
WORKDIR /JupyterNotes

# Copy over python scripts
COPY train3.py /JupyterNotes
COPY convnet3.py /JupyterNotes
COPY dataset2.py /JupyterNotes

RUN mkdir /JupyterNotes/red.png && mkdir /JupyterNotes/green.png
COPY red.png /JupyterNotes/red.png
COPY green.png /JupyterNotes/green.png


EXPOSE 8080

RUN cd /JupyterNotes

# Run command to keep container running
CMD cd ; jupyter notebook --port=8080 --no-browser --ip=0.0.0.0 --allow-root
