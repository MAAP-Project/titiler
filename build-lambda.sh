#!/bin/sh -e

rm -rf ./lambda ./lambda.zip

echo "installing packages..."
pip install --upgrade pip

# pin rasterio, morecantile, rio-tiler, and
# cogeo-mosaic to fix issue with openssl initialization failing, 
# causing 500 responses. Rasterio <1.3 seemed to be the culprit for that,
# and rasterio 1.3 has better packages for curl, openssl, and/or gdal
# pin markupsafe so jinja will have access to soft_unicode
# https://github.com/aws/aws-sam-cli/issues/3661#issuecomment-1044340547

python -m pip install \
  --no-cache-dir \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --only-binary=:all: \
  --upgrade \
  --target=./lambda/ \
  ./src/titiler/core \
  ./src/titiler/extensions["cogeo,stac"] \
  ./src/titiler/mosaic \
  ./src/titiler/application \
  mangum==0.17.0 \
  rasterio==1.3.10 \
  morecantile==5.3.0 \
  rio-tiler==6.6.1 \
  cogeo-mosaic==7.1.0 \
  markupsafe==2.0.1 \
  boto3==1.28.27 \
  botocore==1.31.59
# pinning boto3 and botocore above even though we're going to remove them to
# prevent pulling in different transitive dependencies

cd lambda

echo "cleaning up..."

# remove boto3 and botocore, because otherwise the lambda zip is too large
# after exploding
find . -type d -a -name 'boto3' -print0 | xargs -0 rm -rf
find . -type d -a -name 'botocore' -print0 | xargs -0 rm -rf

# remove cache and test files
find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
find . -type d -a -name 'tests' -print0 | xargs -0 rm -rf

# DON'T remove dist-info, because email-validator relies on it for loading
# its version
# find . -type d -a -name '*.dist-info' -print0 | xargs -0 rm -rf

echo "copying handler..."
cp ../deployment/aws/lambda/handler.py .

echo "creating zip..."
zip --quiet -r ../lambda.zip .
cd -
ls -hal ./lambda.zip
