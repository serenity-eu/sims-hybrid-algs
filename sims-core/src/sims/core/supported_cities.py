from pathlib import Path

import geopandas
from geopandas import GeoDataFrame

from .data_providers import up42_provider
from .types import SupportedCity

_ASSETS_DIR = Path(__file__).parent / "assets"


def _read_aoi_for_city(city: SupportedCity) -> GeoDataFrame:
    city_aoi_path = _ASSETS_DIR / f"{str(city)}_300" / "geodata" / "aoi.geojson"
    return geopandas.read_file(city_aoi_path)


def _read_image_set_for_city(city: SupportedCity) -> GeoDataFrame:
    city_image_set_path = (
        _ASSETS_DIR / f"{str(city)}_300" / "geodata" / "original_images.geojson"
    )
    raw_gdf = geopandas.read_file(city_image_set_path)
    # Tokyo Bay data is already normalized (has snake_case columns)
    if city == SupportedCity.TOKYO_BAY:
        return raw_gdf
    return up42_provider.normalize(raw_gdf)


SUPPORTED_CITIES_BOUNDS: dict[SupportedCity, GeoDataFrame] = {
    city: _read_aoi_for_city(city) for city in SupportedCity
}

SUPPORTED_CITIES_IMAGE_SETS: dict[SupportedCity, GeoDataFrame] = {
    city: _read_image_set_for_city(city) for city in SupportedCity
}

SUPPROTED_CITIES_BEST_CRS: dict[SupportedCity, str] = {
    SupportedCity.LAGOS_NIGERIA: 'EPSG:32631',
    SupportedCity.MEXICO_CITY: 'EPSG:32614',
    SupportedCity.PARIS: 'EPSG:2154',
    SupportedCity.RIO_DE_JANEIRO: 'EPSG:32723',
    SupportedCity.TOKYO_BAY: 'EPSG:2459',
}
