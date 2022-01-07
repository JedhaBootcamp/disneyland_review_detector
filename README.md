# Disneyland Review Analyser 

Welcome to our Disneyland Review Analyser. Our algorithm's goal is to analyse french reviews and detect how many "stars" this review would corresponds to. Your task is to:

* Build a docker image that will standardize your working environment. 
    * If you are stuck, you can use: `jedha/tensorflow-simple-image`
* Complete the code within `train.py` file to make it work
* Create an `MLproject` file 
* Run the code locally or remotely (however you prefer)

---

## OPTIONAL - Run GPU on EC2 with Docker 

If you want to run this code on EC2 GPU instances, here is a reminder of the steps to follow:

1. Choose Deep Learning AMI - EC2 with GPU 
    * Deep Learning AMI (Amazon Linux 2) Version 56.0 - ami-0afac37ebdacee753

2. install `mlflow` --> `pip install mlflow`

3. Run `mlflow run GITHUB_URL` with the following paramaters:
    * `-A gpus=all`
    * `-A runtime=nvidia`
    * `-P epochs=XXX`

## Troubleshooting 

* Verify if docker with gpu works:
    * `sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`


Happy Coding! 👩‍💻
