# Release Notes

## 0.14.0-1.0.5 (2024-05-31)

- Mosaic: Added MOSAIC_TILE_TIMEOUT to define maximum rendering time for a single tile.
- Reworked lambda zip as it was too large to pin dependencies and remove boto3 and botocore.

## 0.14.0-1.0.4 (2023-10-04)

- Mosaic: For float-valued data (e.g., COP DEM), rescale before tiling.
- Mosaic: Re-added parameters to mosaic render_tile that were previously removed inadvertently, which broke multi-band raster tiling.

## 0.14.0-1.0.3 (2023-09-12)

- Mosaic: Add POST and OPTIONS as CORS methods allowed

## 0.14.0-1.0.2 (2023-08-16)

- updated from upstream v0.14.0

## 0.12.0-1.0.1 (2023-08-16)

- add boto3 package to lambda zip to satisfy use of dynamodb for mosaic

## 0.12.0-1.0.0 (2023-08-15)

- introduce new split versioning scheme that represents the upstream titiler version (v0.12.0) and the mosaic version (1.0.0)
- fix broken packaging in lambda zip file introduced from upgrading to python 3.10

## 0.12.0 (2023-08-03)

- updated from upstream v0.12.0

## 0.11.7 (2023-07-25)

- updated from upstream v0.11.7
