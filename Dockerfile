FROM python:3.10-slim
WORKDIR /app
RUN mkdir src
RUN mkdir final_sensor_models
ADD ./src/traffic_status_API.py /app/src
ADD ./src/solution.py /app/src
ADD ./final_sensor_models /app/final_sensor_models
ADD ./requirements.txt /app


RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "src/traffic_status_API.py"]
