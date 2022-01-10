# Base image
FROM python:3.7-slim
# FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
# For using anibali...for GPU remove torch from requirements

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*



# Copy relevant files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/

# Set working directory
WORKDIR /

# Insntall python packages. No chache for save space
RUN pip install -r requirements.txt --no-cache-dir

# Set command to run when the image is executed
# '-u' redirects the logs to the console
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]


# docker build -f train.dockerfile . -t trainer:latest
# Build the docker
