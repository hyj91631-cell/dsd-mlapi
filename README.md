## Deploy ML Model as API for Inference with FastAPI
### Requirements
1. Install Docker.
2. Create virtual environment
```
python -m venv venv
```
3. Activate virtual environment
* Windows
```
venv\Scripts\activate
```
* Linux/MacOS
```
source venv/bin/activate
```
4. Install required Python packages:
```
pip install -r requirements.txt
```

### Step 1: Train and Save Model
1. Run the training module to train a used car price prediction model:
```
python -m src.training --data_path data/cars.csv
```
Sample training script: [training.py](src/training.py)  
Sample model registry script: [model_registry.py](src/model_registry.py)

### Step 2: Design and Develop Inference API
1. Using FastAPI, develop an API that takes new data for inference as input and outputs predicted value at endpoint '/api/v1/used-car/price'.
2. Implement ASGI server using Uvicorn for the APIs.  

Sample API script: [app.py](src/app.py)  
Sample Server script: [server.py](server.py)

### Step 3: Test API
Start the API service:
```
python server.py
```
Use Swagger UI (http://127.0.0.1:8000/docs) to test the response of the APIs.

### Step 4: Serving Predictions from Docker
Start Docker engine.  

Next, in command prompt or terminal, navigate to the project directory that contains Dockerfile.  

Build the docker image using:
```
docker build -t ml-fastapi-docker:1.0 .
```

After the building the image, start an application container:
```
docker run --rm -d -p 80:8000 ml-fastapi-docker:1.0
```
Verify whether the container is running:
```
docker ps
```





