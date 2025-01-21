# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Install pipenv
RUN pip install pipenv

# Set a working directory to copy files
WORKDIR /app

# Copy only the Pipfile and Pipfile.lock to take advantage of Docker caching
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install only production dependencies (with --deploy to ensure strict dependency resolution)
RUN pipenv install --system --deploy

# Copy the rest of the application files
COPY scripts /app/scripts
COPY data/processed/data_for_model.csv /app/data/processed
COPY models/best_model.keras /app/models/
COPY models/column_transformer.pkl /app/models/
COPY models/label_encoder.pkl /app/models/
COPY models/feature_info.npy /app/models/

# Expose port for the application
EXPOSE 9696

# Use Gunicorn to serve the app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
