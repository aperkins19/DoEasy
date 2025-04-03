# Welcome

Welcome to DoEasy, an open-source, free to use Design of Experiments platform.
Please feel free to use for academic purposes and cite this repo if you do.
Contact for commerical use: alexperkssynbio@gmail.com

## Installation and Usage


### Docker
You need to have Git, Docker & docker compose installed on your system.

If you've never installed Docker, use this tutorial.
https://www.youtube.com/watch?v=_9AWYlt86B8


### Clone this repo.

`git clone https://github.com/aperkins19/DoEasy.git`

### Build and Run The Container with Docker Compose

1. Enter the directory: `cd DoEasy`

2. Build and run: `sudo docker compose up` or `sudo docker compose up &`

n.b. First, docker will build the image, this may take a few minutes. When it is running, you may have to open a new terminal window for the next steps.

3. Enter the container: `sudo docker exec -it doeasy /bin/bash`

4. From within the docker container, run software:
`bash platform.sh`