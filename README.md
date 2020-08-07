# Instructions for Cell Counting Container
##### Maintainer: C.Deligiorgi2@newcastle.ac.uk

This container uses python:3.6-stretch as a base and Python3 with the model's dependencies installed. The Cell Counting container includes the train3.py model as well as the VGG model. It also has Jupyter Notebooks installed. For a docker container to keep running the Docker Engine needs to be running. If all processes in a docker stop, the container will stop. The container is running by the Jupyter Notebooks service. To access the Cell Counting Container:

* Install Docker Desktop from https://www.docker.com/products/docker-desktop.
Docker Desktop for Windows offers the GUI which allows the user to start, pause and stop the container without the command prompt.

- Use the command prompt for the following:

* Once the Docker Desktop is installed, run the ```docker version``` command which shows the version of Docker installed.

* From the Docker Hub repository https://hub.docker.com/repository/docker/pikachris/cellcounting, copy the docker pull command ``` docker pull pikachris/cellcounting ``` to pull the cellcounting image with all the dependences. The latest tag of the image is 0.2. When downloading a new version of the image, Docker will download only the dependences - layers that are not already exist on the cache.

* Once the image pull is completed, run the following command at the command prompt:

```docker run -p 8080:8080 pikachris/cellcounting:0.2```

This command creates an instance of the container.

* To access Jupyter Notebooks in this container, open a browser and enter http://localhost:8080. This assumes that the docker container is running on the same computer as the browser. If the container is running on another computer on the network, replace localhost with the IP address of that computer. The password for the notebook is ```docker```.

* To stop the docker container right click on Docker dashboard from the system tray, and stop the container from the GUI. Alternatively, type the following command at the command prompt: ```docker stop cellcounting```

* To start the container again, in case of Docker Desktop for Windows, right click on Docker dashboard from the system tray, and start the container from the GUI. Alternatively,  type the following command at the command prompt: ```docker start cellcounting```

* To stop the docker container right click on Docker dashboard from the system tray, and stop the container from the GUI. Alternatively, type the following command at the command prompt: ```docker stop cellcounting```

This docker container uses a volume, notebooks created and saved within the Jupyter interface will persist.

* It is also possible to run the docker image as an ephemeral container. That means the container is temporary and get deleted when it is stopped. Thus, it does not take up resources while it is not running. To run an ephemeral container use the following command: ```docker run --rm -d --name cellcounting -p 8080:8080 -v JupyterNotes:/JupyterNotes cellcounting:0.2```. The --rm command deletes the container when it is stopped.

* Docker Desktop for Windows, allows the user to bind mount directories from the local machine into Docker containers. To do so follow

Docker system tray -> Settings -> Resources -> File Sharing and add the local file to mount into Docker.
