FROM transformer-deploy-base

WORKDIR /transformer_deploy

COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_gpu.txt ./requirements_gpu.txt
COPY ./src/__init__.py ./src/__init__.py
#COPY ./src/transformer_deploy/__init__.py ./src/transformer_deploy/__init__.py
COPY ./src/transformer_deploy ./src/transformer_deploy
COPY ./ ./

RUN pip3 install ".[GPU]" --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir
