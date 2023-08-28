<p align="center">
  <a href="https://github.com/element84/titiler-mosaicjson/actions?query=workflow%3ACI" target="_blank">
      <img src="https://github.com/element84/titiler-mosaicjson/workflows/CI/badge.svg" alt="Test">
  </a>
  <a href="https://codecov.io/gh/element84/titiler-mosaicjson" target="_blank">
      <img src="https://codecov.io/gh/element84/titiler-mosaicjson/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/element84/titiler-mosaicjson/blob/main/LICENSE" target="_blank">
      <img src="https://img.shields.io/github/license/element84/titiler-mosaicjson.svg" alt="Downloads">
  </a>
</p>

---

# titiler-mosaicjson

`titiler-mosaicjson` is a fork of the popular [TiTiler](https://github.com/developmentseed/titiler) dynamic tiling server, derived
from the [NASA IMPACT TiTiler fork](https://github.com/NASA-IMPACT/titiler). This project combines the
the ability to create a virtual mosaic, stored in DynamoDB, from a STAC API search from the NASA IMPACT fork with the latest
TiTiler code.

The upstream TiTiler documentation can be found [here](https://devseed.com/titiler/).

## Installation

To install from sources and run for development:

```shell
git clone https://github.com/element84/titiler-mosaicjson.git
cd titiler

python -m pip install -U pip
python -m pip install -e src/titiler/core -e src/titiler/extensions -e src/titiler/mosaic -e src/titiler/application
python -m pip install uvicorn

uvicorn titiler.application.main:app --reload
```

## Contribution & Development

See [CONTRIBUTING.md](https://github.com/element84/titiler-mosaicjson/blob/main/CONTRIBUTING.md)

To update from upstream:

1. Merge from upstream
2. Preserve README.md, CHANGES.md, and LICENSE
3. Update upstream README.md into README_upstream.md, CHANGES.md into CHANGES_upstream.md,
   and LICENSE into UPSTREAM_LICENSE

## License

See [LICENSE](https://github.com/element84/titiler-mosaicjson/blob/main/LICENSE) and
[UPSTREAM_LICENSE](https://github.com/element84/titiler-mosaicjson/blob/main/LICENSE).

In release v0.12.0-1.0.1, this code was relicensed from MIT to Apache 2.

## Authors

Maintained by [Element 84](<http://element84.com>)

Originially created by [Development Seed](<http://developmentseed.org>)

See [contributors](https://github.com/element84/titiler-mosaicjson/graphs/contributors) for a listing of individual contributors.

## Changes

See [CHANGES.md](https://github.com/element84/titiler-mosaicjson/blob/main/CHANGES.md).

For upstream changes, see [CHANGES_upstream.md](https://github.com/element84/titiler-mosaicjson/blob/main/CHANGES_upstream.md).
