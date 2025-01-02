# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install pipenv
RUN pip install pipenv

# Set a working directory to copy files
WORKDIR /app

# Copy only the Pipfile and Pipfile.lock to take advantage of Docker caching
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install only production dependencies (with --deploy to ensure strict dependency resolution)
RUN pipenv install --system --deploy

# Copy the rest of the application files
COPY predict.py /app/
COPY models/final_model_pipeline.pkl /app/models/
COPY models/dtypes.pkl /app/models/
COPY models/column_names.pkl /app/models/
COPY data/processed/teams_stats_2024.csv /app/data/processed/
COPY scripts/teams_data.py /app/scripts/

# Expose port for the application
EXPOSE 9696

# Use Gunicorn to serve the app
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]