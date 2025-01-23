# Public Docker image for the Lambda function
FROM public.ecr.aws/lambda/python:3.10

# Install keras-tuner not pipenv
RUN pip install pipenv

# Set a working directory to copy files
WORKDIR /app

# Copy only the Pipfile and Pipfile.lock to take advantage of Docker caching
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install only production dependencies (with --deploy to ensure strict dependency resolution)
RUN pipenv install --system --deploy

# Copy the rest of the application files
COPY scripts/predict.py /app/scripts/predict.py
COPY data/processed/data_for_model.csv /app/data/processed/data_for_model.csv
COPY models /app/models/

# Set the PYTHONPATH so that Gunicorn can find the modules correctly
ENV PYTHONPATH=/app

# Expose port for the application
EXPOSE 9696

# Use Gunicorn to serve the app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "scripts.predict:app"]
