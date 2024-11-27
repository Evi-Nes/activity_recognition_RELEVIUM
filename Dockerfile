FROM tensorflow/tensorflow:2.18.0
ENV PYTHONUNBUFFERED=1

RUN pip install --break-system-packages influxdb-client==1.47.0 pandas==2.2.3 scikit-learn==1.5.2 pymongo==4.10.1
RUN pip install python-dotenv==1.0.1 mongoengine==0.29.1 falcon==4.0.2 falcon-multipart==0.2.0

COPY . /code
WORKDIR /code

EXPOSE 4001
CMD ["python", "/code/consumer/app.py"]
