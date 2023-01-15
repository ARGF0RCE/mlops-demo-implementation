# lightweight python
FROM python:3.10.9-slim

RUN apt-get update

# Copy local code to container image
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies -r requirements.txt
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD ["streamlit","run", "--server.enableCORS", "false", "imgwebapp.py"]