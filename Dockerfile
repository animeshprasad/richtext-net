FROM ubuntu
MAINTAINER noviscl

# TO BE UPDATED
# Run apt to install OS packages

RUN apt update

RUN apt install -y  python3.6  python-pip git 
#python-setuptools python-dev build-essential

RUN pip install ipython matplotlib numpy pandas scikit-learn scipy six

RUN pip install git+https://www.github.com/keras-team/keras-contrib.git

RUN pip install absl-py==0.6.1 astor==0.7.1 backports.weakref==1.0.post1 beautifulsoup4==4.6.3 boto==2.49.0 boto3==1.9.45 botocore==1.12.45 bz2file==0.98 certifi==2018.10.15 chardet==3.0.4 decorator==4.3.0 docutils==0.14 enum34==1.1.6 funcsigs==1.0.2 futures==3.2.0 gast==0.2.0 gensim==3.6.0 grpcio==1.16.1 h5py==2.8.0 idna==2.7 jmespath==0.9.3 Keras==2.2.4 Keras-Applications==1.0.6 Keras-Preprocessing==1.0.5 Markdown==3.0.1 mock==2.0.0 networkx==2.2 nltk==3.3 numpy==1.15.4 pandas==0.23.4 pbr==5.1.1 protobuf==3.6.1 python-crfsuite==0.9.6 python-dateutil==2.7.5 pytz==2018.7 PyYAML==3.13 requests==2.20.1 s3transfer==0.1.13 scikit-learn==0.20.1 scipy==1.1.0 six==1.11.0 sklearn==0.0 sklearn-crfsuite==0.3.6 smart-open==1.7.1 tabulate==0.8.2 tensorboard==1.12.0 tensorflow==1.12.0 termcolor==1.1.0 tqdm==4.28.1 urllib3==1.24.1 Werkzeug==0.14.1 wikipedia==1.4.0

#RUN mkdir /work


#RUN git clone https://github.com/animeshprasad/richtext-ptr-net.git ./project/richtext-ptr-net

