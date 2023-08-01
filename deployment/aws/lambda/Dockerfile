ARG PYTHON_VERSION=3.10

FROM --platform=linux/amd64 public.ecr.aws/lambda/python:${PYTHON_VERSION}

WORKDIR /tmp

COPY titiler.*.tar.gz /tmp/

RUN pip install pip -U
RUN pip install \
        # mangum is required by our handler.py, and isn't a dependency of the
        # titiler packages
        mangum==0.17.0 \
        # pin rasterio, morecantile, rio-tiler, and
        # cogeo-mosaic to fix issue with openssl initialization failing, 
        # causing 500 responses. Rasterio <1.3 seemed to be the culprit for that,
        # and rasterio 1.3 has better packages for curl, openssl, and/or gdal
        rasterio==1.3.6 \
        morecantile==4.3.0 \
        rio-tiler==5.0.3 \
        cogeo-mosaic==6.2.0 \
        # pin markupsafe so jinja will have access to soft_unicode
        # https://github.com/aws/aws-sam-cli/issues/3661#issuecomment-1044340547
        markupsafe==2.0.1 \
        titiler.application==0.12.0 \
        titiler.mosaic==0.12.0 \
        titiler.core==0.12.0 \
        -t /var/task --no-binary pydantic
        # original code also had --no-binary numpy, which takes 20-30 minutes to build

# Reduce package size and remove useless files
RUN cd /var/task && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
RUN cd /var/task && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN cd /var/task && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /var/task -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /var/task/numpy/doc/ /var/task/uvicorn /var/task/boto3* /var/task/botocore* \
        /var/task/bin /var/task/geos_license /var/task/Misc

COPY lambda/handler.py /var/task/handler.py

RUN mkdir "/asset-output"

CMD ["echo", "hello world"]
