"""TiTiler.mosaic Router factories."""

import asyncio
import logging
import os
import time
import uuid
from asyncio import wait_for
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import urlencode

import morecantile
import rasterio
from cogeo_mosaic.backends import BaseBackend, DynamoDBBackend, MosaicBackend
from cogeo_mosaic.errors import MosaicError
from cogeo_mosaic.models import Info as mosaicInfo
from cogeo_mosaic.mosaic import MosaicJSON
from fastapi import Depends, Header, HTTPException, Path, Query
from geojson_pydantic.features import Feature
from geojson_pydantic.geometries import Polygon
from morecantile import tms as morecantile_tms
from morecantile.defaults import TileMatrixSets
from pydantic import conint
from pystac_client import Client
from rio_tiler.constants import MAX_THREADS, WGS84_CRS
from rio_tiler.io import BaseReader, COGReader, MultiBandReader, MultiBaseReader, Reader
from rio_tiler.models import Bounds
from rio_tiler.mosaic.methods import PixelSelectionMethod
from rio_tiler.mosaic.methods.base import MosaicMethodBase
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from typing_extensions import Annotated

from titiler.core.dependencies import (
    BufferParams,
    ColorFormulaParams,
    CoordCRSParams,
    DefaultDependency,
)
from titiler.core.factory import BaseTilerFactory, img_endpoint_params
from titiler.core.models.mapbox import TileJSON
from titiler.core.resources.enums import ImageType, MediaType, OptionalHeader
from titiler.core.resources.responses import GeoJSONResponse, JSONResponse, XMLResponse
from titiler.core.utils import render_image
from titiler.mosaic.models.responses import Point
from titiler.mosaic.resources.models import (
    Link,
    MosaicEntity,
    StacApiQueryRequestBody,
    StoreException,
    TooManyResultsException,
    UnsupportedOperationException,
    UrisRequestBody,
)

from .settings import MosaicSettings


# This code is copied from marblecutter
#  https://github.com/mojodna/marblecutter/blob/master/marblecutter/stats.py
# License:
# Original work Copyright 2016 Stamen Design
# Modified work Copyright 2016-2017 Seth Fitzsimmons
# Modified work Copyright 2016 American Red Cross
# Modified work Copyright 2016-2017 Humanitarian OpenStreetMap Team
# Modified work Copyright 2017 Mapzen
class Timer(object):
    """Time a code block."""

    def __enter__(self):
        """Starts timer."""
        self.start = time.time()
        return self

    def __exit__(self, ty, val, tb):
        """Stops timer."""
        self.end = time.time()
        self.elapsed = self.end - self.start

    @property
    def from_start(self):
        """Return time elapsed from start."""
        return time.time() - self.start


def PixelSelectionParams(
    pixel_selection: Annotated[  # type: ignore
        Literal[tuple([e.name for e in PixelSelectionMethod])],
        Query(description="Pixel selection method."),
    ] = "first",
) -> MosaicMethodBase:
    """
    Returns the mosaic method used to combine datasets together.
    """
    return PixelSelectionMethod[pixel_selection].value()


@dataclass
class MosaicTilerFactory(BaseTilerFactory):
    """
    MosaicTiler Factory.

    The main difference with titiler.endpoint.factory.TilerFactory is that this factory
    needs the `reader` to be of `cogeo_mosaic.backends.BaseBackend` type (e.g MosaicBackend) and a `dataset_reader` (BaseReader).
    """

    reader: Type[BaseBackend] = MosaicBackend
    dataset_reader: Union[
        Type[BaseReader],
        Type[MultiBaseReader],
        Type[MultiBandReader],
    ] = Reader

    backend_dependency: Type[DefaultDependency] = DefaultDependency

    pixel_selection_dependency: Callable[..., MosaicMethodBase] = PixelSelectionParams

    supported_tms: TileMatrixSets = morecantile_tms
    default_tms: str = "WebMercatorQuad"

    # Add/Remove some endpoints
    add_viewer: bool = True

    logger = logging.getLogger(__name__)

    def register_routes(self):
        """
        This Method register routes to the router.

        Because we wrap the endpoints in a class we cannot define the routes as
        methods (because of the self argument). The HACK is to define routes inside
        the class method and register them after the class initialization.

        """

        self.read()
        self.bounds()
        self.info()
        self.tile()
        self.tilejson()
        self.wmts()
        self.point()
        self.validate()
        self.assets()
        self.mosaics()

        # Optional Routes
        if self.add_viewer:
            self.map_viewer()

    ############################################################################
    # /read
    ############################################################################
    def read(self):
        """Register / (Get) Read endpoint."""

        @self.router.get(
            "/",
            response_model=MosaicJSON,
            response_model_exclude_none=True,
            responses={200: {"description": "Return MosaicJSON definition"}},
        )
        def read(
            src_path=Depends(self.path_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Read a MosaicJSON"""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return src_dst.mosaic_def

    ############################################################################
    # /bounds
    ############################################################################
    def bounds(self):
        """Register /bounds endpoint."""

        @self.router.get(
            "/bounds",
            response_model=Bounds,
            responses={200: {"description": "Return the bounds of the MosaicJSON"}},
        )
        def bounds(
            src_path=Depends(self.path_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return the bounds of the MosaicJSON."""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return {"bounds": src_dst.geographic_bounds}

    ############################################################################
    # /info
    ############################################################################
    def info(self):
        """Register /info endpoint"""

        @self.router.get(
            "/info",
            response_model=mosaicInfo,
            responses={200: {"description": "Return info about the MosaicJSON"}},
        )
        def info(
            src_path=Depends(self.path_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return basic info."""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return src_dst.info()

        @self.router.get(
            "/info.geojson",
            response_model=Feature[Polygon, mosaicInfo],
            response_model_exclude_none=True,
            response_class=GeoJSONResponse,
            responses={
                200: {
                    "content": {"application/geo+json": {}},
                    "description": "Return mosaic's basic info as a GeoJSON feature.",
                }
            },
        )
        def info_geojson(
            src_path=Depends(self.path_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return mosaic's basic info as a GeoJSON feature."""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    info = src_dst.info()
                    return Feature(
                        type="Feature",
                        geometry=Polygon.from_bounds(*info.bounds),
                        properties=info,
                    )

    ############################################################################
    # /tiles
    ############################################################################
    def tile(self):  # noqa: C901
        """Register /tiles endpoints."""

        @self.router.get("/tiles/{z}/{x}/{y}", **img_endpoint_params)
        @self.router.get("/tiles/{z}/{x}/{y}.{format}", **img_endpoint_params)
        @self.router.get("/tiles/{z}/{x}/{y}@{scale}x", **img_endpoint_params)
        @self.router.get("/tiles/{z}/{x}/{y}@{scale}x.{format}", **img_endpoint_params)
        @self.router.get("/tiles/{tileMatrixSetId}/{z}/{x}/{y}", **img_endpoint_params)
        @self.router.get(
            "/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}", **img_endpoint_params
        )
        @self.router.get(
            "/tiles/{tileMatrixSetId}/{z}/{x}/{y}@{scale}x", **img_endpoint_params
        )
        @self.router.get(
            "/tiles/{tileMatrixSetId}/{z}/{x}/{y}@{scale}x.{format}",
            **img_endpoint_params,
        )
        def tile(
            z: Annotated[
                int,
                Path(
                    description="Identifier (Z) selecting one of the scales defined in the TileMatrixSet and representing the scaleDenominator the tile.",
                ),
            ],
            x: Annotated[
                int,
                Path(
                    description="Column (X) index of the tile on the selected TileMatrix. It cannot exceed the MatrixHeight-1 for the selected TileMatrix.",
                ),
            ],
            y: Annotated[
                int,
                Path(
                    description="Row (Y) index of the tile on the selected TileMatrix. It cannot exceed the MatrixWidth-1 for the selected TileMatrix.",
                ),
            ],
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                f"Identifier selecting one of the TileMatrixSetId supported (default: '{self.default_tms}')",
            ] = self.default_tms,
            scale: Annotated[
                conint(gt=0, le=4),
                "Tile size scale. 1=256x256, 2=512x512...",
            ] = 1,
            format: Annotated[
                ImageType,
                "Default will be automatically defined if the output image needs a mask (png) or not (jpeg).",
            ] = None,
            src_path=Depends(self.path_dependency),
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            pixel_selection=Depends(self.pixel_selection_dependency),
            buffer=Depends(BufferParams),
            post_process=Depends(self.process_dependency),
            rescale=Depends(self.rescale_dependency),
            color_formula=Depends(ColorFormulaParams),
            colormap=Depends(self.colormap_dependency),
            render_params=Depends(self.render_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Create map tile from a COG."""
            if scale < 1 or scale > 4:
                raise HTTPException(
                    400,
                    f"Invalid 'scale' parameter: {scale}. Scale HAVE TO be between 1 and 4",
                )

            threads = int(os.getenv("MOSAIC_CONCURRENCY", MAX_THREADS))

            strict_zoom = str(os.getenv("MOSAIC_STRICT_ZOOM", False)).lower() in [
                "true",
                "yes",
            ]

            tms = self.supported_tms.get(tileMatrixSetId)
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    tms=tms,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:

                    if strict_zoom and (z < src_dst.minzoom or z > src_dst.maxzoom):
                        raise HTTPException(
                            400,
                            f"Invalid ZOOM level {z}. Should be between {src_dst.minzoom} and {src_dst.maxzoom}",
                        )

                    image, assets = src_dst.tile(
                        x,
                        y,
                        z,
                        pixel_selection=pixel_selection,
                        tilesize=scale * 256,
                        threads=threads,
                        buffer=buffer,
                        **layer_params,
                        **dataset_params,
                    )

            if post_process:
                image = post_process(image)

            if rescale:
                image.rescale(rescale)

            if color_formula:
                image.apply_color_formula(color_formula)

            content, media_type = render_image(
                image,
                output_format=format,
                colormap=colormap,
                **render_params,
            )

            headers: Dict[str, str] = {}
            if OptionalHeader.x_assets in self.optional_headers:
                headers["X-Assets"] = ",".join(assets)

            return Response(content, media_type=media_type, headers=headers)

    def tilejson(self):  # noqa: C901
        """Add tilejson endpoint."""

        @self.router.get(
            "/tilejson.json",
            response_model=TileJSON,
            responses={200: {"description": "Return a tilejson"}},
            response_model_exclude_none=True,
        )
        @self.router.get(
            "/{tileMatrixSetId}/tilejson.json",
            response_model=TileJSON,
            responses={200: {"description": "Return a tilejson"}},
            response_model_exclude_none=True,
        )
        def tilejson(
            request: Request,
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                f"Identifier selecting one of the TileMatrixSetId supported (default: '{self.default_tms}')",
            ] = self.default_tms,
            src_path=Depends(self.path_dependency),
            tile_format: Annotated[
                Optional[ImageType],
                Query(
                    description="Default will be automatically defined if the output image needs a mask (png) or not (jpeg).",
                ),
            ] = None,
            tile_scale: Annotated[
                int,
                Query(
                    gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
                ),
            ] = 1,
            minzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default minzoom."),
            ] = None,
            maxzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default maxzoom."),
            ] = None,
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            pixel_selection=Depends(self.pixel_selection_dependency),
            buffer=Depends(BufferParams),
            post_process=Depends(self.process_dependency),
            rescale=Depends(self.rescale_dependency),
            color_formula=Depends(ColorFormulaParams),
            colormap=Depends(self.colormap_dependency),
            render_params=Depends(self.render_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return TileJSON document for a COG."""
            route_params = {
                "z": "{z}",
                "x": "{x}",
                "y": "{y}",
                "scale": tile_scale,
                "tileMatrixSetId": tileMatrixSetId,
            }
            if tile_format:
                route_params["format"] = tile_format.value
            tiles_url = self.url_for(request, "tile", **route_params)

            qs_key_to_remove = [
                "tilematrixsetid",
                "tile_format",
                "tile_scale",
                "minzoom",
                "maxzoom",
            ]
            qs = [
                (key, value)
                for (key, value) in request.query_params._list
                if key.lower() not in qs_key_to_remove
            ]
            if qs:
                tiles_url += f"?{urlencode(qs)}"

            tms = self.supported_tms.get(tileMatrixSetId)
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    tms=tms,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    center = list(src_dst.mosaic_def.center)
                    if minzoom is not None:
                        center[-1] = minzoom

                    return {
                        "bounds": src_dst.bounds,
                        "center": tuple(center),
                        "minzoom": minzoom if minzoom is not None else src_dst.minzoom,
                        "maxzoom": maxzoom if maxzoom is not None else src_dst.maxzoom,
                        "tiles": [tiles_url],
                    }

    def map_viewer(self):  # noqa: C901
        """Register /map endpoint."""

        @self.router.get("/map", response_class=HTMLResponse)
        @self.router.get("/{tileMatrixSetId}/map", response_class=HTMLResponse)
        def map_viewer(
            request: Request,
            src_path=Depends(self.path_dependency),
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                f"Identifier selecting one of the TileMatrixSetId supported (default: '{self.default_tms}')",
            ] = self.default_tms,
            tile_format: Annotated[
                Optional[ImageType],
                Query(
                    description="Default will be automatically defined if the output image needs a mask (png) or not (jpeg).",
                ),
            ] = None,
            tile_scale: Annotated[
                int,
                Query(
                    gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
                ),
            ] = 1,
            minzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default minzoom."),
            ] = None,
            maxzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default maxzoom."),
            ] = None,
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            pixel_selection=Depends(self.pixel_selection_dependency),
            buffer=Depends(BufferParams),
            rescale=Depends(self.rescale_dependency),
            color_formula=Depends(ColorFormulaParams),
            colormap=Depends(self.colormap_dependency),
            render_params=Depends(self.render_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return TileJSON document for a dataset."""
            tilejson_url = self.url_for(
                request, "tilejson", tileMatrixSetId=tileMatrixSetId
            )
            if request.query_params._list:
                tilejson_url += f"?{urlencode(request.query_params._list)}"

            tms = self.supported_tms.get(tileMatrixSetId)
            return self.templates.TemplateResponse(
                name="map.html",
                context={
                    "request": request,
                    "tilejson_endpoint": tilejson_url,
                    "tms": tms,
                    "resolutions": [tms._resolution(matrix) for matrix in tms],
                },
                media_type="text/html",
            )

    def wmts(self):  # noqa: C901
        """Add wmts endpoint."""

        @self.router.get("/WMTSCapabilities.xml", response_class=XMLResponse)
        @self.router.get(
            "/{tileMatrixSetId}/WMTSCapabilities.xml", response_class=XMLResponse
        )
        def wmts(
            request: Request,
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                f"Identifier selecting one of the TileMatrixSetId supported (default: '{self.default_tms}')",
            ] = self.default_tms,
            src_path=Depends(self.path_dependency),
            tile_format: Annotated[
                ImageType,
                Query(description="Output image type. Default is png."),
            ] = ImageType.png,
            tile_scale: Annotated[
                int,
                Query(
                    gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
                ),
            ] = 1,
            minzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default minzoom."),
            ] = None,
            maxzoom: Annotated[
                Optional[int],
                Query(description="Overwrite default maxzoom."),
            ] = None,
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            pixel_selection=Depends(self.pixel_selection_dependency),
            buffer=Depends(BufferParams),
            post_process=Depends(self.process_dependency),
            rescale=Depends(self.rescale_dependency),
            color_formula=Depends(ColorFormulaParams),
            colormap=Depends(self.colormap_dependency),
            render_params=Depends(self.render_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """OGC WMTS endpoint."""
            route_params = {
                "z": "{TileMatrix}",
                "x": "{TileCol}",
                "y": "{TileRow}",
                "scale": tile_scale,
                "format": tile_format.value,
                "tileMatrixSetId": tileMatrixSetId,
            }
            tiles_url = self.url_for(request, "tile", **route_params)

            qs_key_to_remove = [
                "tilematrixsetid",
                "tile_format",
                "tile_scale",
                "minzoom",
                "maxzoom",
                "service",
                "request",
            ]
            qs = [
                (key, value)
                for (key, value) in request.query_params._list
                if key.lower() not in qs_key_to_remove
            ]
            if qs:
                tiles_url += f"?{urlencode(qs)}"

            tms = self.supported_tms.get(tileMatrixSetId)
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    tms=tms,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    bounds = src_dst.geographic_bounds
                    minzoom = minzoom if minzoom is not None else src_dst.minzoom
                    maxzoom = maxzoom if maxzoom is not None else src_dst.maxzoom

            tileMatrix = []
            for zoom in range(minzoom, maxzoom + 1):  # type: ignore
                matrix = tms.matrix(zoom)
                tm = f"""
                        <TileMatrix>
                            <ows:Identifier>{matrix.id}</ows:Identifier>
                            <ScaleDenominator>{matrix.scaleDenominator}</ScaleDenominator>
                            <TopLeftCorner>{matrix.pointOfOrigin[0]} {matrix.pointOfOrigin[1]}</TopLeftCorner>
                            <TileWidth>{matrix.tileWidth}</TileWidth>
                            <TileHeight>{matrix.tileHeight}</TileHeight>
                            <MatrixWidth>{matrix.matrixWidth}</MatrixWidth>
                            <MatrixHeight>{matrix.matrixHeight}</MatrixHeight>
                        </TileMatrix>"""
                tileMatrix.append(tm)

            return self.templates.TemplateResponse(
                "wmts.xml",
                {
                    "request": request,
                    "tiles_endpoint": tiles_url,
                    "bounds": bounds,
                    "tileMatrix": tileMatrix,
                    "tms": tms,
                    "title": "Mosaic",
                    "layer_name": "mosaic",
                    "media_type": tile_format.mediatype,
                },
                media_type=MediaType.xml.value,
            )

    ############################################################################
    # /point (Optional)
    ############################################################################
    def point(self):
        """Register /point endpoint."""

        @self.router.get(
            "/point/{lon},{lat}",
            response_model=Point,
            response_class=JSONResponse,
            responses={200: {"description": "Return a value for a point"}},
        )
        def point(
            response: Response,
            lon: Annotated[float, Path(description="Longitude")],
            lat: Annotated[float, Path(description="Latitude")],
            src_path=Depends(self.path_dependency),
            coord_crs=Depends(CoordCRSParams),
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Get Point value for a Mosaic."""
            threads = int(os.getenv("MOSAIC_CONCURRENCY", MAX_THREADS))

            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    values = src_dst.point(
                        lon,
                        lat,
                        coord_crs=coord_crs or WGS84_CRS,
                        threads=threads,
                        **layer_params,
                        **dataset_params,
                    )

            return {
                "coordinates": [lon, lat],
                "values": [
                    (src, pts.data.tolist(), pts.band_names) for src, pts in values
                ],
            }

    def validate(self):
        """Register /validate endpoint."""

        @self.router.post("/validate")
        def validate(body: MosaicJSON):
            """Validate a MosaicJSON"""
            return True

    ############################################################################
    # /mosaics
    ############################################################################
    def mosaics(self):  # noqa: C901
        """Register /mosaics endpoints."""

        # with dynamodb backend, the tiles field for this is always empty
        # https://github.com/developmentseed/cogeo-mosaic/issues/175
        @self.router.get(
            "/mosaics/{mosaic_id}",
            response_model=MosaicEntity,
            responses={
                HTTP_200_OK: {
                    "description": "Return a Mosaic resource for the given ID."
                },
                HTTP_404_NOT_FOUND: {
                    "description": "Mosaic resource for the given ID does not exist."
                },
            },
        )
        async def get_mosaic(
            request: Request,
            mosaic_id: str,
            env=Depends(self.environment_dependency),
            reader_params=Depends(self.reader_dependency),
        ) -> MosaicEntity:
            self_uri = self.url_for(request, "get_mosaic", mosaic_id=mosaic_id)
            with rasterio.Env(**env):
                if await retrieve(mosaic_id, reader_params):
                    return mk_mosaic_entity(mosaic_id=mosaic_id, self_uri=self_uri)
                else:
                    raise HTTPException(
                        HTTP_404_NOT_FOUND,
                        "Error: mosaic with given ID does not exist.",
                    )

        @self.router.get(
            "/mosaics/{mosaic_id}/mosaicjson",
            response_model=MosaicJSON,
            responses={
                200: {
                    "description": "Return a MosaicJSON definition for the given ID."
                },
                404: {
                    "description": "Mosaic resource for the given ID does not exist."
                },
            },
        )
        async def get_mosaic_mosaicjson(
            mosaic_id: str,
            env=Depends(self.environment_dependency),
            reader_params=Depends(self.reader_dependency),
        ) -> MosaicJSON:
            with rasterio.Env(**env):
                if m := await retrieve(mosaic_id, reader_params, include_tiles=True):
                    return m
                else:
                    raise HTTPException(
                        HTTP_404_NOT_FOUND,
                        "Error: mosaic with given ID does not exist.",
                    )

        # derived from cogeo.xyz
        @self.router.get(
            r"/mosaics/{mosaic_id}/tilejson.json",
            response_model=TileJSON,
            responses={
                200: {"description": "Return a tilejson for the given ID."},
                404: {
                    "description": "Mosaic resource for the given ID does not exist."
                },
            },
            response_model_exclude_none=True,
        )
        async def get_mosaic_tilejson(
            mosaic_id: str,
            request: Request,
            tile_format: Optional[ImageType] = Query(
                None, description="Output image type. Default is auto."
            ),
            tile_scale: int = Query(
                1, gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
            ),
            minzoom: Optional[int] = Query(
                None, description="Overwrite default minzoom."
            ),
            maxzoom: Optional[int] = Query(
                None, description="Overwrite default maxzoom."
            ),
            layer_params=Depends(self.layer_dependency),  # noqa
            dataset_params=Depends(self.dataset_dependency),  # noqa
            render_params=Depends(self.render_dependency),  # noqa
            colormap=Depends(self.colormap_dependency),  # noqa,
            env=Depends(self.environment_dependency),
            reader_params=Depends(self.reader_dependency),
        ) -> TileJSON:
            """Return TileJSON document for a MosaicJSON."""

            kwargs = {
                "mosaic_id": mosaic_id,
                "z": "{z}",
                "x": "{x}",
                "y": "{y}",
                "scale": tile_scale,
            }
            if tile_format:
                kwargs["format"] = tile_format.value
            tiles_url = self.url_for(request, "tile", **kwargs)

            q = dict(request.query_params)
            q.pop("TileMatrixSetId", None)
            q.pop("tile_format", None)
            q.pop("tile_scale", None)
            qs = urlencode(list(q.items()))
            tiles_url += f"?{qs}"

            with rasterio.Env(**env):
                if mosaicjson := await retrieve(mosaic_id, reader_params):
                    center = list(mosaicjson.center)
                    if minzoom:
                        center[-1] = minzoom
                    return TileJSON(
                        bounds=mosaicjson.bounds,
                        center=tuple(center),
                        minzoom=minzoom if minzoom is not None else mosaicjson.minzoom,
                        maxzoom=maxzoom if maxzoom is not None else mosaicjson.maxzoom,
                        name=mosaic_id,
                        tiles=[tiles_url],
                    )
                else:
                    raise HTTPException(
                        HTTP_404_NOT_FOUND,
                        "Error: mosaic with given ID does not exist.",
                    )

        @self.router.post(
            "/mosaics",
            status_code=HTTP_201_CREATED,
            responses={
                HTTP_201_CREATED: {"description": "Created a new mosaic"},
                HTTP_409_CONFLICT: {
                    "description": "Conflict while trying to create mosaic"
                },
                HTTP_500_INTERNAL_SERVER_ERROR: {
                    "description": "Mosaic could not be created"
                },
            },
            response_model=MosaicEntity,
        )
        async def post_mosaics(
            request: Request,
            response: Response,
            content_type: Optional[str] = Header(None),
            env=Depends(self.environment_dependency),
        ) -> MosaicEntity:
            """Create a MosaicJSON"""

            mosaicjson = await populate_mosaicjson(request, content_type)
            mosaic_id = str(uuid.uuid4())

            # duplicate IDs are unlikely to exist, but handle it just to be safe
            try:
                with rasterio.Env(**env):
                    await store(mosaic_id, mosaicjson, env, overwrite=False)
            except StoreException as e:
                raise HTTPException(
                    HTTP_409_CONFLICT, "Error: mosaic with given ID already exists"
                ) from e
            except Exception as e:
                logging.error(f"could not save mosaic: {e}")
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR, "Error: could not save mosaic"
                ) from e

            self_uri = self.url_for(request, "get_mosaic", mosaic_id=mosaic_id)

            response.headers["Location"] = self_uri

            return mk_mosaic_entity(mosaic_id, self_uri)

        # @self.router.put(
        #     "/mosaics/{mosaic_id}",
        #     status_code=HTTP_204_NO_CONTENT,
        #     responses={
        #         HTTP_204_NO_CONTENT: {"description": "Updated a mosaic"},
        #         HTTP_404_NOT_FOUND: {"description": "Mosaic with ID not found"},
        #         HTTP_500_INTERNAL_SERVER_ERROR: {
        #             "description": "Mosaic could not be updated"
        #         },
        #     },
        # )
        # async def put_mosaic(
        #     mosaic_id: str,
        #     request: Request,
        #     content_type: Optional[str] = Header(None),
        # ) -> None:
        #     """Update an existing MosaicJSON"""

        #     if not await retrieve(mosaic_id):
        #         raise HTTPException(
        #             HTTP_404_NOT_FOUND, "Error: mosaic with given ID does not exist."
        #         )

        #     try:
        #         mosaicjson = await populate_mosaicjson(request, content_type)
        #         await store(mosaic_id, mosaicjson, overwrite=True)
        #     except StoreException:
        #         raise HTTPException(
        #             HTTP_404_NOT_FOUND, "Error: mosaic with given ID does not exist."
        #         )
        #     except Exception:
        #         raise HTTPException(
        #             HTTP_500_INTERNAL_SERVER_ERROR, "Error: could not update mosaic."
        #         )
        #
        #     return

        # note: cogeo-mosaic doesn't clear the cache on write/delete, so these will stay until the TTL expires
        # https://github.com/developmentseed/cogeo-mosaic/issues/176
        # @self.router.delete(
        #     "/mosaics/{mosaic_id}", status_code=HTTP_204_NO_CONTENT,
        # )
        # async def delete_mosaic(mosaic_id: str) -> None:
        #     """Delete an existing MosaicJSON"""

        #     if not await retrieve(mosaic_id):
        #         raise HTTPException(
        #             HTTP_404_NOT_FOUND, "Error: mosaic with given ID does not exist."
        #         )

        #     try:
        #         await delete(mosaic_id)
        #     except UnsupportedOperationException:
        #         raise HTTPException(
        #             HTTP_405_METHOD_NOT_ALLOWED,
        #             "Error: mosaic with given ID cannot be deleted because the datastore does not support it.",
        #         )

        # derived from cogeo-xyz
        @self.router.get(
            r"/mosaics/{mosaic_id}/tiles/{z}/{x}/{y}", **img_endpoint_params
        )
        @self.router.get(
            r"/mosaics/{mosaic_id}/tiles/{z}/{x}/{y}.{format}", **img_endpoint_params
        )
        @self.router.get(
            r"/mosaics/{mosaic_id}/tiles/{z}/{x}/{y}@{scale}x", **img_endpoint_params
        )
        @self.router.get(
            r"/mosaics/{mosaic_id}/tiles/{z}/{x}/{y}@{scale}x.{format}",
            **img_endpoint_params,
        )
        async def tile(
            mosaic_id: str,
            z: int = Path(..., ge=0, le=30, description="Mercator tiles's zoom level"),
            x: int = Path(..., description="Mercator tiles's column"),
            y: int = Path(..., description="Mercator tiles's row"),
            scale: Annotated[
                conint(gt=0, lt=4), "Tile size scale. 1=256x256, 2=512x512..."
            ] = 1,
            format: Annotated[
                ImageType,
                "Output image type. Default is auto.",
            ] = None,
            layer_params=Depends(self.layer_dependency),
            dataset_params=Depends(self.dataset_dependency),
            render_params=Depends(self.render_dependency),
            colormap=Depends(self.colormap_dependency),
            pixel_selection: PixelSelectionMethod = Query(
                PixelSelectionMethod.first, description="Pixel selection method."
            ),
            env=Depends(self.environment_dependency),
            reader_params=Depends(self.reader_dependency),
            rescale=Depends(self.rescale_dependency),
        ):
            """Create map tile from a mosaic."""

            try:
                with rasterio.Env(**env):
                    (content, data_assets, img_format, timings) = await wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None,  # executor
                            render_tile,  # func
                            mk_src_path(mosaic_id),
                            z,
                            x,
                            y,
                            scale,
                            format,
                            layer_params,
                            dataset_params,
                            render_params,
                            colormap,
                            pixel_selection,
                            reader_params,
                            rescale,
                        ),
                        None,
                    )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR,
                    "Error: timeout executing rendering tile.",
                ) from e

            headers: Dict[str, str] = {}

            if OptionalHeader.server_timing in self.optional_headers:
                headers["Server-Timing"] = ", ".join(
                    [f"{name};dur={time}" for (name, time) in timings]
                )

            if OptionalHeader.x_assets in self.optional_headers:
                headers["X-Assets"] = ",".join(data_assets)

            return Response(content, media_type=img_format.mediatype, headers=headers)

        @self.router.get(
            "/mosaics/{mosaic_id}/WMTSCapabilities.xml", response_class=XMLResponse
        )
        def wmts(
            request: Request,
            mosaic_id: str,
            tile_format: ImageType = Query(
                ImageType.png, description="Output image type. Default is png."
            ),
            tile_scale: int = Query(
                1, gt=0, lt=4, description="Tile size scale. 1=256x256, 2=512x512..."
            ),
            minzoom: Optional[int] = Query(
                None, description="Overwrite default minzoom."
            ),
            maxzoom: Optional[int] = Query(
                None, description="Overwrite default maxzoom."
            ),
            layer_params=Depends(self.layer_dependency),  # noqa
            dataset_params=Depends(self.dataset_dependency),  # noqa
            render_params=Depends(self.render_dependency),  # noqa
            colormap=Depends(self.colormap_dependency),  # noqa
            pixel_selection: PixelSelectionMethod = Query(
                PixelSelectionMethod.first, description="Pixel selection method."
            ),  # noqa
        ):
            """OGC WMTS endpoint."""

            tiles_url = self.url_for(
                request,
                "tile",
                **{
                    "mosaic_id": mosaic_id,
                    "z": "{TileMatrix}",
                    "x": "{TileCol}",
                    "y": "{TileRow}",
                    "scale": tile_scale,
                    "format": tile_format.value,
                },
            )

            q = dict(request.query_params)
            q.pop("tile_format", None)
            q.pop("tile_scale", None)
            q.pop("minzoom", None)
            q.pop("maxzoom", None)
            q.pop("SERVICE", None)
            q.pop("REQUEST", None)
            qs = urlencode(list(q.items()))
            tiles_url += f"?{qs}"

            mosaic_uri = mk_src_path(mosaic_id)

            with self.reader(mosaic_uri) as src_dst:
                bounds = src_dst.bounds
                minzoom = minzoom if minzoom is not None else src_dst.minzoom
                maxzoom = maxzoom if maxzoom is not None else src_dst.maxzoom

            tms = morecantile.tms.get("WebMercatorQuad")

            tileMatrix = []
            for zoom in range(minzoom, maxzoom + 1):
                matrix = tms.matrix(zoom)
                tm = f"""
                        <TileMatrix>
                            <ows:Identifier>{matrix.id}</ows:Identifier>
                            <ScaleDenominator>{matrix.scaleDenominator}</ScaleDenominator>
                            <TopLeftCorner>{matrix.pointOfOrigin[0]} {matrix.pointOfOrigin[1]}</TopLeftCorner>
                            <TileWidth>{matrix.tileWidth}</TileWidth>
                            <TileHeight>{matrix.tileHeight}</TileHeight>
                            <MatrixWidth>{matrix.matrixWidth}</MatrixWidth>
                            <MatrixHeight>{matrix.matrixHeight}</MatrixHeight>
                        </TileMatrix>"""
                tileMatrix.append(tm)

            return self.templates.TemplateResponse(
                "wmts.xml",
                {
                    "request": request,
                    "tiles_endpoint": tiles_url,
                    "bounds": bounds,
                    "tileMatrix": tileMatrix,
                    "tms": tms,
                    "title": "Cloud Optimized GeoTIFF",
                    "layer_name": "cogeo",
                    "media_type": tile_format.mediatype,
                },
                media_type=MediaType.xml.value,
            )

        ###################

        #####################
        # auxiliary methods #
        #####################

        def render_tile(
            mosaic_uri: str,
            z: int,
            x: int,
            y: int,
            scale: int,
            format: ImageType,
            layer_params,
            dataset_params,
            render_params,
            colormap,
            pixel_selection: PixelSelectionMethod,
            reader_params,
            rescale,
        ) -> Tuple[bytes, Any, ImageType, List[Tuple[str, float]]]:
            """Create map tile from a COG."""
            timings = []

            tilesize = scale * 256

            threads = int(os.getenv("MOSAIC_CONCURRENCY", MAX_THREADS))
            with Timer() as t:
                with self.reader(
                    mosaic_uri,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                ) as src_dst:
                    mosaic_read = t.from_start
                    timings.append(("mosaicread", round(mosaic_read * 1000, 2)))

                    data, _ = src_dst.tile(
                        x,
                        y,
                        z,
                        pixel_selection,
                        threads=threads,
                        tilesize=tilesize,
                        **layer_params,
                        **dataset_params,
                    )
            timings.append(("dataread", round((t.elapsed - mosaic_read) * 1000, 2)))

            if not format:
                format = ImageType.jpeg if data.mask.all() else ImageType.png

            with Timer() as t:
                image = data.post_process()
            timings.append(("postprocess", round(t.elapsed * 1000, 2)))

            if rescale:
                image.rescale(rescale)

            with Timer() as t:
                content = image.render(
                    img_format=format.driver,
                    colormap=colormap,
                    **format.profile,
                    **render_params,
                )
            timings.append(("format", round(t.elapsed * 1000, 2)))

            return content, data.assets, format, timings

        async def mosaicjson_from_urls(urisrb: UrisRequestBody) -> MosaicJSON:

            if len(urisrb.urls) > MAX_ITEMS:
                raise HTTPException(
                    HTTP_400_BAD_REQUEST,
                    f"Error: a maximum of {MAX_ITEMS} URLs can be mosaiced.",
                )

            try:
                mosaicjson = await wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,  # executor
                        lambda: MosaicJSON.from_urls(
                            urls=urisrb.urls,
                            minzoom=urisrb.minzoom,
                            maxzoom=urisrb.maxzoom,
                            max_threads=int(
                                os.getenv("MOSAIC_CONCURRENCY", MAX_THREADS)
                            ),  # todo
                        ),
                    ),
                    20,
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR,
                    "Error: timeout reading URLs and generating MosaicJSON definition",
                ) from e

            if mosaicjson is None:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR,
                    "Error: could not extract mosaic data",
                )

            mosaicjson.name = urisrb.name
            mosaicjson.description = urisrb.description
            mosaicjson.attribution = urisrb.attribution
            mosaicjson.version = urisrb if urisrb.version else "0.0.1"

            return mosaicjson

        async def mosaicjson_from_stac_api_query(  # noqa: C901
            req: StacApiQueryRequestBody,
        ) -> MosaicJSON:
            """Create a mosaic for the given parameters"""

            if not req.stac_api_root:
                raise HTTPException(
                    HTTP_400_BAD_REQUEST,
                    "Error: stac_api_root field must be non-empty.",
                )

            try:
                try:
                    features = await wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, execute_stac_search, req  # executor  # func
                        ),
                        30,
                    )
                except asyncio.TimeoutError as e:
                    raise HTTPException(
                        HTTP_500_INTERNAL_SERVER_ERROR,
                        "Error: timeout executing STAC API search.",
                    ) from e
                except TooManyResultsException as e:
                    raise HTTPException(
                        HTTP_400_BAD_REQUEST,
                        f"Error: too many results from STAC API Search: {e}",
                    ) from e

                if not features:
                    raise HTTPException(
                        HTTP_500_INTERNAL_SERVER_ERROR,
                        "Error: STAC API Search returned no results.",
                    )

                try:
                    mosaicjson = await wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None,
                            extract_mosaicjson_from_features,
                            features,
                            req.asset_name if req.asset_name else "visual",
                        ),
                        60,  # todo: how much time should/can it take?
                    )
                except asyncio.TimeoutError as e:
                    raise HTTPException(
                        HTTP_500_INTERNAL_SERVER_ERROR,
                        "Error: timeout reading a COG asset and generating MosaicJSON definition",
                    ) from e

                if mosaicjson is None:
                    raise HTTPException(
                        HTTP_500_INTERNAL_SERVER_ERROR,
                        "Error: could not extract mosaic data",
                    )

                mosaicjson.name = req.name
                mosaicjson.description = req.description
                mosaicjson.attribution = req.attribution
                mosaicjson.version = req if req.version else "0.0.1"

                return mosaicjson

            except HTTPException as e:
                raise e
            except Exception as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR, f"Error: {e}"
                ) from e

        MAX_ITEMS = 1000

        def execute_stac_search(mosaic_request: StacApiQueryRequestBody) -> List[dict]:
            try:
                search_result = Client.open(mosaic_request.stac_api_root).search(
                    ids=mosaic_request.ids,
                    collections=mosaic_request.collections,
                    datetime=mosaic_request.datetime,
                    bbox=mosaic_request.bbox,
                    intersects=mosaic_request.intersects,
                    query=mosaic_request.query,
                    max_items=mosaic_request.max_items
                    if mosaic_request.max_items and mosaic_request.max_items < MAX_ITEMS
                    else MAX_ITEMS,
                    limit=mosaic_request.limit if mosaic_request.limit else 100,
                )
                # matched = search_result.matched()
                # if matched > MAX_ITEMS:
                #     raise TooManyResultsException(
                #         f"too many results: {matched} Items matched, but only a maximum of {MAX_ITEMS} are allowed."
                #     )

                return list(search_result.items_as_dicts())
            except TooManyResultsException as e:
                raise e
            except Exception as e:
                raise Exception(f"STAC Search error: {e}") from e

        # assumes all assets are uniform. get the min and max zoom from the first.
        def extract_mosaicjson_from_features(
            features: List[dict], asset_name: str
        ) -> Optional[MosaicJSON]:
            if features:
                try:
                    with COGReader(asset_href(features[0], asset_name)) as cog:
                        info = cog.info()
                    return MosaicJSON.from_features(
                        features,
                        minzoom=info.minzoom,
                        maxzoom=info.maxzoom,
                        accessor=partial(asset_href, asset_name=asset_name),
                    )

                # when Item geometry is a MultiPolygon (instead of a Polygon), supermercado raises
                # handle error "local variable 'x' referenced before assignment"
                # supermercado/burntiles.py ", line 38, in _feature_extrema
                # as this method only handles Polygon, LineString, and Point :grimace:
                # https://github.com/mapbox/supermercado/issues/47
                except UnboundLocalError as e:
                    raise Exception(
                        "STAC Items likely have MultiPolygon geometry, and only Polygon is supported."
                    ) from e
                except Exception as e:
                    raise Exception(
                        f"Error extracting mosaic data from results: {e}"
                    ) from e
            else:
                return None

        # todo: make this safer in case visual doesn't exist
        # how to handle others?
        # support for selection by role?
        def asset_href(feature: dict, asset_name: str) -> str:
            if href := feature.get("assets", {}).get(asset_name, {}).get("href"):
                return href
            else:
                raise Exception(f"Asset with name '{asset_name}' could not be found.")

        def mk_src_path(mosaic_id: str) -> str:
            mosaic_settings = MosaicSettings()
            if mosaic_settings.backend == "dynamodb://":
                return f"{mosaic_settings.backend}{mosaic_settings.host}:{mosaic_id}"
            else:
                return f"{mosaic_settings.backend}{mosaic_settings.host}/{mosaic_id}{mosaic_settings.format}"

        async def store(
            mosaic_id: str, mosaicjson: MosaicJSON, env, overwrite: bool
        ) -> None:
            try:
                existing = await retrieve(mosaic_id, env)
            except Exception:
                existing = False

            if not overwrite and existing:
                raise StoreException("Attempting to create already existing mosaic")
            if overwrite and not existing:
                raise StoreException("Attempting to update non-existant mosaic")

            mosaic_uri = mk_src_path(mosaic_id)

            try:
                await wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,  # executor
                        mosaic_write,  # func
                        mosaic_uri,
                        mosaicjson,
                        overwrite,
                    ),
                    20,
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR,
                    "Error: timeout storing mosaic in datastore",
                ) from e

        def mosaic_write(
            mosaic_uri: str, mosaicjson: MosaicJSON, overwrite: bool
        ) -> None:
            with self.reader(mosaic_uri, mosaic_def=mosaicjson) as mosaic:
                mosaic.write(overwrite=overwrite)

        async def retrieve(
            mosaic_id: str, reader_params, include_tiles: bool = False
        ) -> Optional[MosaicJSON]:
            mosaic_uri = mk_src_path(mosaic_id)

            try:
                return await wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None,  # executor
                        read_mosaicjson_sync,  # func
                        mosaic_uri,
                        reader_params,
                        include_tiles,
                    ),
                    20,
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR,
                    "Error: timeout retrieving mosaic from datastore.",
                ) from e
            except MosaicError:
                return None

        def read_mosaicjson_sync(
            mosaic_uri: str, reader_params, include_tiles: bool
        ) -> MosaicJSON:

            with self.reader(
                mosaic_uri,
                reader=self.dataset_reader,
                reader_options={**reader_params},
            ) as mosaic:
                mosaicjson = mosaic.mosaic_def
                if include_tiles and isinstance(mosaic, DynamoDBBackend):
                    keys = (mosaic._fetch_dynamodb(qk) for qk in mosaic._quadkeys)
                    mosaicjson.tiles = {x["quadkey"]: x["assets"] for x in keys}
                return mosaicjson

        async def delete(mosaic_id: str) -> None:
            mosaic_uri = mk_src_path(mosaic_id)

            try:
                await wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, delete_mosaicjson_sync, mosaic_uri  # executor  # func
                    ),
                    20,
                )
            except asyncio.TimeoutError as e:
                raise HTTPException(
                    HTTP_500_INTERNAL_SERVER_ERROR, "Error: timeout deleting mosaic."
                ) from e

            return

        def delete_mosaicjson_sync(mosaic_uri: str) -> None:
            with self.reader(
                mosaic_uri,
                reader=self.dataset_reader,
                **self.backend_options,
            ) as mosaic:
                if isinstance(mosaic, DynamoDBBackend):
                    mosaic.delete()  # delete is only supported by DynamoDB
                else:
                    raise UnsupportedOperationException("Delete is not supported")

        def mk_mosaic_entity(mosaic_id, self_uri):
            return MosaicEntity(
                id=mosaic_id,
                links=[
                    Link(
                        rel="self", href=self_uri, type="application/json", title="Self"
                    ),
                    Link(
                        rel="mosaicjson",
                        href=f"{self_uri}/mosaicjson",
                        type="application/json",
                        title="MosiacJSON",
                    ),
                    Link(
                        rel="tilejson",
                        href=f"{self_uri}/tilejson.json",
                        type="application/json",
                        title="TileJSON",
                    ),
                    Link(
                        rel="tiles",
                        href=f"{self_uri}/tiles/{{z}}/{{x}}/{{y}}",
                        type="application/json",
                        title="Tiles",
                    ),
                    Link(
                        rel="wmts",
                        href=f"{self_uri}/WMTSCapabilities.xml",
                        type="application/json",
                        title="WMTS",
                    ),
                ],
            )

        async def populate_mosaicjson(request, content_type):
            body_json = await request.json()
            if (
                not content_type
                or content_type == "application/json"
                or content_type == "application/json; charset=utf-8"
                or content_type == "application/vnd.titiler.mosaicjson+json"
            ):
                mosaicjson = MosaicJSON(**body_json)
            elif content_type == "application/vnd.titiler.urls+json":
                mosaicjson = await mosaicjson_from_urls(UrisRequestBody(**body_json))
            elif content_type == "application/vnd.titiler.stac-api-query+json":
                mosaicjson = await mosaicjson_from_stac_api_query(
                    StacApiQueryRequestBody(**body_json)
                )
            else:
                raise HTTPException(
                    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    "Error: media in Content-Type header is not supported.",
                )
            return mosaicjson

    def assets(self):
        """Register /assets endpoint."""

        @self.router.get(
            "/{minx},{miny},{maxx},{maxy}/assets",
            responses={200: {"description": "Return list of COGs in bounding box"}},
        )
        def assets_for_bbox(
            minx: Annotated[float, Path(description="Bounding box min X")],
            miny: Annotated[float, Path(description="Bounding box min Y")],
            maxx: Annotated[float, Path(description="Bounding box max X")],
            maxy: Annotated[float, Path(description="Bounding box max Y")],
            src_path=Depends(self.path_dependency),
            coord_crs=Depends(CoordCRSParams),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return a list of assets which overlap a bounding box"""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return src_dst.assets_for_bbox(
                        minx,
                        miny,
                        maxx,
                        maxy,
                        coord_crs=coord_crs or WGS84_CRS,
                    )

        @self.router.get(
            "/{lon},{lat}/assets",
            responses={200: {"description": "Return list of COGs"}},
        )
        def assets_for_lon_lat(
            lon: Annotated[float, Path(description="Longitude")],
            lat: Annotated[float, Path(description="Latitude")],
            src_path=Depends(self.path_dependency),
            coord_crs=Depends(CoordCRSParams),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return a list of assets which overlap a point"""
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return src_dst.assets_for_point(
                        lon,
                        lat,
                        coord_crs=coord_crs or WGS84_CRS,
                    )

        @self.router.get(
            "/{z}/{x}/{y}/assets",
            responses={200: {"description": "Return list of COGs"}},
        )
        @self.router.get(
            "/{tileMatrixSetId}/{z}/{x}/{y}/assets",
            responses={200: {"description": "Return list of COGs"}},
        )
        def assets_for_tile(
            z: Annotated[
                int,
                Path(
                    description="Identifier (Z) selecting one of the scales defined in the TileMatrixSet and representing the scaleDenominator the tile.",
                ),
            ],
            x: Annotated[
                int,
                Path(
                    description="Column (X) index of the tile on the selected TileMatrix. It cannot exceed the MatrixHeight-1 for the selected TileMatrix.",
                ),
            ],
            y: Annotated[
                int,
                Path(
                    description="Row (Y) index of the tile on the selected TileMatrix. It cannot exceed the MatrixWidth-1 for the selected TileMatrix.",
                ),
            ],
            tileMatrixSetId: Annotated[
                Literal[tuple(self.supported_tms.list())],
                f"Identifier selecting one of the TileMatrixSetId supported (default: '{self.default_tms}')",
            ] = self.default_tms,
            src_path=Depends(self.path_dependency),
            backend_params=Depends(self.backend_dependency),
            reader_params=Depends(self.reader_dependency),
            env=Depends(self.environment_dependency),
        ):
            """Return a list of assets which overlap a given tile"""
            tms = self.supported_tms.get(tileMatrixSetId)
            with rasterio.Env(**env):
                with self.reader(
                    src_path,
                    tms=tms,
                    reader=self.dataset_reader,
                    reader_options={**reader_params},
                    **backend_params,
                ) as src_dst:
                    return src_dst.assets_for_tile(x, y, z)
