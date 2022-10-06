# FROM python:3.10.7-slim
# FROM python:3.9.5-slim-buster
# FROM python:3.8.12-slim 
FROM python:3.7.12-buster


ENV PYTHONUNBUFFERED 1

WORKDIR /app
RUN apt-get update && apt-get install --yes build-essential


RUN apt-get install -y python-tk python3-tk tk-dev
RUN pip install --upgrade pip
# RUN pip install --upgrade protobuf
RUN pip install tk


# # Install Tkinter
# # RUN apt-get install tk -y
# RUN apt-get install --yes tk-dev

# # cf https://stackoverflow.com/questions/63590165/running-tkinter-on-docker-container
# # RUN apt install python3-tk

# #sinon tester https://stackoverflow.com/questions/63590165/running-tkinter-on-docker-container
# # RUN apk update && apk add tk

ADD requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
ADD app.py ./app.py
ADD style.css ./style.css
ADD load_css.py ./load_css.py
# ADD v4_best_incl_timestamp ./v4_best_incl_timestamp
ADD df.csv ./df.csv

COPY .streamlit ./.streamlit
COPY secrets ./secrets
COPY src ./src
COPY images ./images
COPY pickled ./pickled
COPY wordclouds ./wordclouds
#Add the necessary files but prefer .parquet format !
#COPY large_def.parquet ./large_def.parquet




EXPOSE 80
CMD [ "streamlit", "run", "app.py" ]