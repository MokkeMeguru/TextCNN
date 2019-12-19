FROM tensorflow/tensorflow:2.0.0-gpu-py3
RUN mkdir /workspace
RUN pip install -U pip
RUN apt-get update & apt-get install graphviz
RUN pip install python-language-server[all]  ptvsd yapf tensorflow-probability sentencepiece pandas tensorflow_datasets pydot
