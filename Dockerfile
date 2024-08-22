# Use the official Ubuntu base image
FROM ubuntu:22.04


# Set working directory
WORKDIR /app


# Install necessary tools (git and any system packages you might need)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository
RUN git clone https://github.com/SimaArasteh/binhuntertool.git

# Set the working directory to the cloned repository
WORKDIR /app/binhuntertool

# Install system packages if specified in package_list.txt
RUN xargs -a package_list.txt apt-get install -y

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt