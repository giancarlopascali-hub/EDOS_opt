# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container, setting ownership to our new user
COPY --chown=user requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container, setting ownership to the user
COPY --chown=user . .

# Switch to the non-root user for security and to comply with HF limits
USER user

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variables
ENV FLASK_APP=app.py
ENV PORT=7860

# Run the application with gunicorn for production reliability
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]
