# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /my_folder
WORKDIR /app

# Copy the requirements file and verything else in this folder:
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# What should be executed when the contrainer is built/started: Run python script:
CMD ["python", "etl.py"]