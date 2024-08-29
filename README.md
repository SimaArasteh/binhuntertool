# binhuntertool

This is an implementation of the paper *BinHunter: A Fine-Grained Graph Representation for Localizing Vulnerabilities in Binary Executables*. BinHunter is a tool designed to find and localize various types of vulnerabilities within binary programs. It leverages a Graph Convolutional Neural Network (GCN) to train on vulnerabilities. For each type of vulnerability, we train a separate binary classifier. During the training phase, BinHunter trains on a subgraph of the Program Dependence Graph (PDG) of a function that locates vulnerability and patch patterns extracted using debug symbols (DWARF) in the binary. In the testing phase, however, it utilizes calls to external functions to slice the PDG.
# requirements

As mentioned in our paper, we performed all experiments on a Linux server with 256 GB RAM and an Intel XeonE5-1650 CPU with 12 cores. 


# Installation

To install all dependencies, first download the Dockerfile from https://drive.google.com/file/d/1NM3f8HvLPMlps4ZDyBqdQEug03Xubae9/view?usp=drive_link.

Then create a directory called bindocker and put the dockerfile inside this directory. 

```
mkdir bindocker
```
Then change your directory to bindocker.

```
cd bindocker
```

Now build the docker using this command.

```
docker build -t binhunter .
```

When the docker is built successfully, then run the docker using the command below.

```
docker run -it binhunter
```

you are now inside the docker. if you type whoami you should see binhuntuser. Now run the commands below to install pip.

```
sudo apt update
sudo apt install python3-pip
```

First give the permission to binhunt user to write and then clone the binhunter repo.

```
sudo chmod 777 .

git clone https://github.com/SimaArasteh/binhuntertool.git
```
Now change directory to binhuntertool. 
```
cd binhuntertool
```
Now install all dependencies and requirements using following command.

```
pip3 install -r requirements.txt

```
When all dependencies installed successfully, download the data from https://drive.google.com/file/d/1NM3f8HvLPMlps4ZDyBqdQEug03Xubae9/view?usp=drive_link. Then copy the data.zip inside the docker. In your local machine find the container id of binhunter. 

```
docker ps
docker cp data.zip container-id:/binhuntertool
```
After copying data into the docker, unzip the data.zip.

```
unzip data.zip -d .
```

# usage

Now it is time to run binhunter. You can run binhunter for each type of vulnerability using following command.
```
python3 run_binhunter.py CWE121 
```

**Note that for each round of running binhunter, we shuffle the training data and each time you run, you get different results but the results should be close to Table3.**