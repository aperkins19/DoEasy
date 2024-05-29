
# Dockerfile 04/2021
# Python environment for exporting data analysis for processing on remote systems.

# Ubuntu version 20.04
FROM ubuntu:20.04

# Overwrites the R installation questions 
ENV DEBIAN_FRONTEND noninteractive

# Specifies R version
ENV R_BASE_VERSION=4.0.0

# Update Ubuntu and install packages
RUN apt-get update && \
    apt-get install -y \
    nano \
    software-properties-common
    
# Installs R
RUN apt-get install -y gnupg2

RUN apt-get install -y libcairo2-dev


# Installs R 
RUN apt-get install -y --no-install-recommends build-essential

# Installs Python
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev

# Upgrades the Python Package Manager
RUN pip3 -q install pip --upgrade
RUN pip3 -q install setuptools
RUN pip3 -q install --upgrade setuptools
RUN apt-get install -y liblzma-dev libbz2-dev libicu-dev libblas-dev


# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Makes a directory in the Ubuntu Root called /app
# Sets it to the directory to work in
# Copies everything over from the current local directory into /app
RUN mkdir /app/
WORKDIR /app/
COPY . .

# Installs the Python packages
RUN pip3 install -r installation/python_requirements.txt

# jq is used to parse json in bash
RUN apt-get update && apt-get clean
RUN apt-get install -y git
RUN apt-get install -y jq

USER root


WORKDIR /app/

# Runs Streamlit Notebook on startup
ENTRYPOINT ["streamlit", "run", "./app.py", "--server.port=7001", "--server.address=0.0.0.0"]