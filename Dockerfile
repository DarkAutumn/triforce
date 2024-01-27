FROM tensorflow/tensorflow:2.13.0-gpu
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 6006
