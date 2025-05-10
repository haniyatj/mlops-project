FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy only the linear regression model
COPY models/linear_regression_model.joblib /app/models/

# Copy API code
COPY api/ /app/

# Set environment variables
ENV MODEL_PATH="/app/models/linear_regression_model.joblib"
ENV PORT=8000
ENV DEBUG=False

# Expose the port
EXPOSE 8000

# Command to run the API with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]