#Download base image python:3.6-stretch
FROM python:3.6-stretch

LABEL maintainer="C.Deligiorgi2@newcastle.ac.uk"

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

# Copy over train3.py scripts
RUN mkdir /JupyterNotes/modelTrain3
COPY modelTrain3/train3.py /JupyterNotes/modelTrain3
COPY modelTrain3/convnet3.py /JupyterNotes/modelTrain3
COPY modelTrain3/dataset2.py /JupyterNotes/modelTrain3
COPY Welcome.ipynb /JupyterNotes

# Copy over VGG model scripts
RUN mkdir /JupyterNotes/VGG
COPY VGG/VGG_train.py /JupyterNotes/VGG
COPY VGG/Conv.py /JupyterNotes/VGG
COPY VGG/dataset2.py /JupyterNotes/VGG


# Copy Data images
RUN mkdir -p /JupyterNotes/Data/healthy && mkdir /JupyterNotes/Data/senescent
ADD Data/healthy /JupyterNotes/Data/healthy
ADD Data/senescent /JupyterNotes/Data/senescent


EXPOSE 8080

RUN cd /JupyterNotes

# Run command to keep container running
CMD cd ; jupyter notebook --port=8080 --no-browser --ip=0.0.0.0 --allow-root