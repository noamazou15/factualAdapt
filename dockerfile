FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/noamazou15/factualAdapt.git .
RUN pip install --no-cache-dir transformers==4.40.2 accelerate==0.27.2 peft==0.11.1 trl==0.8.6 bitsandbytes==0.43.1 mlflow==2.14.1 --ignore-installed blinker

COPY main.py .
ENV PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python", "run_experiments.py"]