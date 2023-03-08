FROM lambci/lambda:build-python3.8

WORKDIR /tmp

COPY titiler.*.tar.gz /tmp/

RUN pip install \
        # mangum > 0.11.x removes the "log_level" parameter from the Mangum constructor
        mangum==0.11.0 \
        # rasterio 1.2.0 wheels are built using GDAL 3.2 and PROJ 7 which we found having a
        # performance downgrade: https://github.com/developmentseed/titiler/discussions/216
        rasterio==1.1.8 \
        # pin markupsafe so jinja will have access to soft_unicode
        # https://github.com/aws/aws-sam-cli/issues/3661#issuecomment-1044340547
        markupsafe==2.0.1 \
        titiler.application-0.3.5.tar.gz titiler.mosaic-0.3.5.tar.gz titiler.core-0.3.5.tar.gz \
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
