FROM lambci/lambda:build-python3.8

WORKDIR /tmp

COPY titiler.*.tar.gz /tmp/

# rasterio 1.2.0 wheels are built using GDAL 3.2 and PROJ 7 which we found having a
# performance downgrade: https://github.com/developmentseed/titiler/discussions/216
RUN pip install \
        titiler.application-0.3.5.tar.gz titiler.mosaic-0.3.5.tar.gz titiler.core-0.3.5.tar.gz \
        mangum>=0.10.0 rasterio==1.1.8 -t /var/task --no-binary numpy,pydantic --no-cache-dir

# Reduce package size and remove useless files
RUN cd /var/task && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
RUN cd /var/task && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN cd /var/task && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /var/task -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /var/task/numpy/doc/
RUN rm -rdf /var/task/uvicorn

COPY lambda/handler.py /var/task/handler.py

RUN mkdir "/asset-output"

CMD ["echo", "hello world"]
