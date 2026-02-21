#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, and parks.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import networkx as nx

from geopandas import GeoDataFrame, GeoSeries
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from networkx import MultiDiGraph
from shapely.geometry import Point, box as shapely_box
from shapely.ops import linemerge, polygonize, unary_union
from tqdm import tqdm

# App modules
from font_management import load_fonts
from rotation_management import rotate_graph_and_features, draw_north_badge
from cache_management import cache_get, cache_set, CacheError

ox.settings.use_cache=True
ox.settings.log_console=False

THEMES_DIR = "themes"
POSTERS_DIR = "posters/perso"

FILE_ENCODING = "utf-8"

# Font loading handled by font_management.py module
FONTS = load_fonts()


def is_latin_script(text):
    """
    Check if text is primarily Latin script.
    Used to determine if letter-spacing should be applied to city names.

    :param text: Text to analyze
    :return: True if text is primarily Latin script, False otherwise
    """
    if not text:
        return True

    latin_count = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            # Latin Unicode ranges:
            # - Basic Latin: U+0000 to U+007F
            # - Latin-1 Supplement: U+0080 to U+00FF
            # - Latin Extended-A: U+0100 to U+017F
            # - Latin Extended-B: U+0180 to U+024F
            if ord(char) < 0x250:
                latin_count += 1

    # If no alphabetic characters, default to Latin (numbers, symbols, etc.)
    if total_alpha == 0:
        return True

    # Consider it Latin if >80% of alphabetic characters are Latin
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        # Fallback to embedded terracotta theme
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "railway": "#FF0000",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }

    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✔ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def get_edge_colors_by_type(g):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []

    for _u, _v, data in g.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')

        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        # Assign color based on road type
        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        elif highway in ["raceway"]:
            color = THEME.get("road_raceway", THEME['road_default'])
        else:
            color = THEME['road_default']

        edge_colors.append(color)

    return edge_colors


def compute_line_width_scale(
    distance_m: float,
    G,                     # the MultiDiGraph from OSMnx
    min_scale: float = 0.55,
    max_scale: float = 1.80,
    reference_distance: float = 12000.0,
) -> float:
    """
    Compute a global line width multiplier based on:
      - Requested distance (larger area → thinner lines by default)
      - Actual graph density (more edges per area → thinner lines to reduce clutter)
    
    Returns value typically in 0.55–1.8 range.
    """
    if G is None or G.number_of_edges() == 0:
        return 1.0  # fallback

    # ── Part 1: distance-based base scale (inverse)
    # smaller radius → expect denser → allow slightly bolder lines
    dist_factor = reference_distance / max(distance_m, 2000.0)
    
    # soften the curve a bit (avoids extreme changes)
    dist_factor = dist_factor ** 0.75
    
    # ── Part 2: real density penalty
    # rough area in km² (very approximate, assumes circular)
    radius_km = distance_m / 1000.0
    approx_area_km2 = 3.1416 * (radius_km ** 2)
    
    edges = G.number_of_edges()
    density = edges / max(approx_area_km2, 0.1)   # edges per km²
    
    # Typical "comfortable" density for posters ~ 800–1800 edges/km²
    # above that → start thinning
    target_density = 1200.0
    density_factor = target_density / max(density, 200.0)
    
    # combine (geometric mean = balanced influence)
    combined = (dist_factor * density_factor) ** 0.5
    
    # final clamping — prevents unusable extremes
    scale = max(min_scale, min(max_scale, combined))
    
    return scale


def get_edge_widths_by_type(g, distance):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    # Calculate the base thickness for a standard residential road
    width_scale = compute_line_width_scale(distance,g)

    edge_width_dict = {
                        "motorway":          1.50 * width_scale,
                        "motorway_link":     1.35 * width_scale,
                        "trunk":             1.15 * width_scale,
                        "primary":           1.00 * width_scale,
                        "secondary":         0.80 * width_scale,
                        "tertiary":          0.60 * width_scale,
                        "residential":       0.40 * width_scale,
                        "living_street":     0.38 * width_scale,
                        "service":           0.32 * width_scale,
                        "unclassified":      0.35 * width_scale,
                        "footway":           0.22 * width_scale,   # if you plot them
                        "path":              0.18 * width_scale,
                        "railway":           0.55 * width_scale,   # if included
                        "raceway":           1.00 * width_scale,   # if included
                        "default":           0.35 * width_scale,
                    }
    
    edge_widths = []

    for u, v, key in g.edges(keys=True):  # or G.edges() if no keys needed
        highway = g[u][v][key].get("highway", "default")  # sometimes it's a list → take first

        if isinstance(highway, list):
            highway = highway[0] if highway else 'default'
        
        width = edge_width_dict.get(highway, edge_width_dict["default"])
        edge_widths.append(width)

    return edge_widths


def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✔ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)

    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running. "
                    "Run this script in a synchronous environment."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✔ Found: {addr}")
        else:
            print("✔ Found location (address not available)")
        print(f"✔ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested radius.
    """
    lat, lon = center_lat_lon

    # Project center point into graph CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    # Start from the *requested* radius
    half_x = dist
    half_y = dist

    # Cut inward to match aspect
    if aspect > 1:  # landscape → reduce height
        half_y = half_x / aspect
    else:  # portrait → reduce width
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def _is_land_polygon(polygon, coastline_geom):
    """
    Determine if a polygon is land using the OSM coastline direction convention.

    In OpenStreetMap, coastlines are oriented with land on the LEFT and water
    on the RIGHT when following the direction of the way. This function checks
    which side of the nearest coastline segment a polygon falls on.

    Args:
        polygon: A Shapely Polygon to classify
        coastline_geom: The projected coastline geometry (LineString or MultiLineString)

    Returns:
        True if the polygon is on the land side, False if water
    """
    test_point = polygon.representative_point()

    # Find the nearest individual LineString segment
    if coastline_geom.geom_type == 'MultiLineString':
        nearest_line = min(coastline_geom.geoms, key=lambda l: l.distance(test_point))
    elif coastline_geom.geom_type == 'LineString':
        nearest_line = coastline_geom
    else:
        return False

    # Project test point onto the nearest coastline
    param = nearest_line.project(test_point)

    # Get local direction of coastline at the nearest point
    epsilon = 1.0  # 1 meter in projected CRS
    p1 = nearest_line.interpolate(max(0, param - epsilon))
    p2 = nearest_line.interpolate(min(nearest_line.length, param + epsilon))

    # Direction vector of coastline
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    # Vector from coastline point to polygon test point
    nearest_point = nearest_line.interpolate(param)
    cx = test_point.x - nearest_point.x
    cy = test_point.y - nearest_point.y

    # Cross product: positive = left side = land, negative = right side = water
    cross = dx * cy - dy * cx
    return cross > 0


def build_sea_polygons(coastline_gdf, g_proj, crop_xlim, crop_ylim, center_lat_lon):
    """
    Build sea/ocean polygons from OSM coastline data.

    In OpenStreetMap, seas and oceans are defined by coastline lines rather
    than water polygons. This function converts coastline lines into renderable
    water polygons by splitting the viewport into land and water regions.

    Uses the OSM coastline direction convention (land on left, water on right)
    to correctly classify all land masses, even when multiple disconnected
    land polygons exist (e.g. Istanbul's European and Asian sides).

    Args:
        coastline_gdf: GeoDataFrame of coastline LineString features (or None)
        g_proj: Projected graph (used for CRS)
        crop_xlim: (xmin, xmax) tuple from get_crop_limits
        crop_ylim: (ymin, ymax) tuple from get_crop_limits
        center_lat_lon: (lat, lon) tuple of the map center

    Returns:
        GeoDataFrame of water polygons in the projected CRS, or None
    """
    if coastline_gdf is None or coastline_gdf.empty:
        return None

    crs = g_proj.graph["crs"]

    # Filter to line geometries only
    line_mask = coastline_gdf.geometry.type.isin(["LineString", "MultiLineString"])
    coast_lines = coastline_gdf[line_mask]
    if coast_lines.empty:
        return None

    # Project coastline to graph CRS
    try:
        coast_proj = ox.projection.project_gdf(coast_lines, to_crs=crs)
    except Exception:
        try:
            coast_proj = coast_lines.to_crs(crs)
        except Exception:
            return None

    # Build viewport rectangle from crop limits
    viewport = shapely_box(crop_xlim[0], crop_ylim[0], crop_xlim[1], crop_ylim[1])

    # Merge coastline fragments and clip to viewport
    merged = linemerge(list(coast_proj.geometry))
    clipped = merged.intersection(viewport)

    if clipped.is_empty:
        return None

    # Combine clipped coastline with viewport boundary to form closed regions
    combined = unary_union([clipped, viewport.boundary])

    # Create polygons from the line network
    polygons = list(polygonize(combined))
    if not polygons:
        return None

    # Classify each polygon using coastline direction convention.
    # OSM coastlines have land on the left, water on the right.
    water_polys = [p for p in polygons if not _is_land_polygon(p, clipped)]

    if not water_polys:
        return None

    return GeoDataFrame(geometry=water_polys, crs=crs)


def fetch_graph(point, dist, ntype='all') -> MultiDiGraph | None:
    """
    Fetch street network graph from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches all network types
    within the specified distance from the center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point

    Returns:
        MultiDiGraph of street network, or None if fetch fails
    """
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✔ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:           
        print(f"\nOSMnx fetching graph: {ntype}")
        g_all = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type=ntype, truncate_by_edge=True)
        
        custom_filter = '["highway"~"raceway|service|track"]'
        g_race = ox.graph_from_point(point, dist=dist, custom_filter=custom_filter)
      
        g = nx.compose(g_all, g_race)
        
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    """
    Fetch geographic features (water, parks, etc.) from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches features matching
    the specified OSM tags within distance from center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point
        tags: Dictionary of OSM tags to filter features
        name: Name for this feature type (for caching and logging)

    Returns:
        GeoDataFrame of features, or None if fetch fails
    """
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✔ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def safe_project(data, target_crs):
    """
    Safely projects either a GeoDataFrame or an OSMnx MultiDiGraph to the target CRS.
    To be used after fetch_features -> GeoDataFrame or after fetch_graph -> MultiDiGraph
    """
    if data is None:
        return None
        
    # Handle GeoDataFrames (as you currently do)
    if isinstance(data, GeoDataFrame):
        if data.empty:
            return None
        try:
            return ox.projection.project_gdf(data, to_crs=target_crs)
        except Exception:
            try:
                return data.to_crs(target_crs)
            except Exception:
                return None

    # Handle MultiDiGraph
    if isinstance(data, (nx.MultiDiGraph, nx.Graph)):
        if not data:  # Tests if the graph has 0 nodes
            return None
        try:
            # OSMnx provides a specific function for projecting graphs
            return ox.projection.project_graph(data, to_crs=target_crs)
        except Exception:
            return None

    return None


def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    display_city=None,
    display_country=None,
    fonts=None,
    fast_mode=False,
    include_railways=False,
    orientation_offset=0.0,
    show_north=False,
):
    """
    Generate a complete map poster with roads, water, parks, and typography.

    Creates a high-quality poster by fetching OSM data, rendering map layers,
    applying the current theme, and adding text labels with coordinates.

    Args:
        city: City name for display on poster
        country: Country name for display on poster
        point: (latitude, longitude) tuple for map center
        dist: Map radius in meters
        output_file: Path where poster will be saved
        output_format: File format ('png', 'svg', or 'pdf')
        width: Poster width in inches (default: 12)
        height: Poster height in inches (default: 16)
        display_country: Optional override for country text on poster
        display_city: Optional override for city name
        fonts:
        fast_mode:
        include_railways: draw railways if defined in theme

    Raises:
        RuntimeError: If street network data cannot be retrieved
    """
    # Handle display names for i18n support
    # Priority: display_city/display_country > city/country
    display_city = display_city or city
    display_country = display_country or country

    print(f"\nGenerating map for {city}, {country}...")

    # -------------
    # 1. Data fetching (with progress bar)
    with tqdm(
        total=6,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        compensated_dist = dist * (max(height, width) / min(height, width))/4 # To compensate for viewport crop

        # Since we are rotating a rectangle, the corners of your poster will "swing" out of the original north-up square. 
        # We need to fetch data using a radius equal to the diagonal of our poster: Fetching 1.5x ensures corners are always filled.
        # Note: The diagonal of a square is ~1.41 times its side (rounded to 1.5)
        fetch_dist = compensated_dist * 1.5

        print(f"\ncompensated dist is: {compensated_dist} - dist is: {fetch_dist}")
        
        # 1.1 Fetch Street Network
        network_type = 'drive' if fast_mode else 'all'
        pbar.set_description(f"Downloading street network ({network_type})")
        g = fetch_graph(point, fetch_dist, network_type)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        # 1.2 Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            fetch_dist,
            #tags={"natural": ["water","bay","strait"], "water": ["river","canal","moat","pond","lake"]},
            tags={
                "natural": ["water", "bay", "strait", "coastline", "sea", "ocean"],
                "waterway": ["riverbank", "river", "canal", "dock", "basin"],
                "place": ["sea", "ocean", "bay"],
                "water": True
            },
            name="water",
        )
        pbar.update(1)
        
		# 1.3 Fetch Coastline
        pbar.set_description("Downloading coastline data")
        coastline = fetch_features(
            point,
            fetch_dist,
            tags={"natural": "coastline"},
            name="coastline",
        )
        pbar.update(1)

        # 1.4 Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            fetch_dist,
            tags={"leisure": "park", "landuse": ["grass", "cemetery"], "natural": "wood"},
            name="parks",
        )
        pbar.update(1)
        
        # 1.5 Fetch airports runways
        pbar.set_description("Downloading remarkable feature spaces")
        runways = fetch_features(
            point,
            fetch_dist,
            tags={"aeroway": ["runway", "taxiway"]},
            name="aeroway",
        )
        pbar.update(1)

        # 1.6 Fetch railways
        if include_railways:
            pbar.set_description("Downloading railways")
            rail_filter = (
                '["railway"~"rail|light_rail|narrow_gauge|monorail|subway|tram|preserved"]'
                '["service"!~"yard|spur"]'
                )
            railways = ox.graph_from_point(point, 
                                fetch_dist,
                                custom_filter=rail_filter,
                                retain_all=True,
                                simplify=False) # Some railways disapear if set to true (ex: Hué, Vietnam)
        else:
            pbar.set_description("Downloading railways skiped")
            railways=None
        pbar.update(1)
    
    print("\n✔ All data retrieved successfully!")

    # -------------
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    print("Projecting features to CRS...")
    g_proj = ox.project_graph(g)

    # Determine cropping limits to maintain the poster aspect ratio
    # We do this early so we can use it for coastline polygonization
    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, compensated_dist)

    # Project everything to the same CRS first
    crs = g_proj.graph['crs']
            
    water_proj = safe_project(water, crs)
    parks_proj = safe_project(parks, crs)
    runways_proj = safe_project(runways, crs)
    railways_proj = safe_project(railways, crs)
    coastline_proj = safe_project(coastline, crs)

    # -------------
    # 3. Rotate everything IN ONE GO
    print("Rotating...")
    features_to_rotate = [water_proj, coastline_proj, parks_proj, runways_proj, railways_proj]
    g_rotated, rotated_list = rotate_graph_and_features(g_proj, features_to_rotate, point, orientation_offset)
    # Re-assign variables from the ROTATED list
    water_rot, coastline_rot, parks_rot, runways_rot, railways_rot = rotated_list

    # -------------
    # 4. Build Sea Polygons using the ROTATED coordinate system and the coastline_rot (which contains the rotated, 
    # un-clipped lines), we ensure the water correctly fills the gaps created by the rotation.
    # Use the original crop_xlim/ylim so the sea fills the final view
    # Note: The sea_polys should be built AFTER the rotation.
    print("Building sea polygons...")
    sea_polys = build_sea_polygons(coastline_rot, g_rotated, crop_xlim, crop_ylim, point)

    edge_colors = get_edge_colors_by_type(g_rotated)
    edge_widths = get_edge_widths_by_type(g_rotated, dist)

    # -------------
    # 5. Plot the ROTATED variables
    print("Plotting...")
    if sea_polys is not None:
        print("....seas/oceans")
        sea_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=0.5)
    if water_rot is not None:
        print("....water")
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots like fountains, private backyard pond,...
        water_polys = water_rot[water_rot.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=0.5)

    # Layer: parks 
    if parks_rot is not None and not parks_rot.empty:
        print("....parks")
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks_rot[parks_rot.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=0.8)

    # Layer: aeroways
    if runways_rot is not None and not runways_rot.empty: 
        # Many airports in OSM are mapped as LineStrings (the centerline of the runway) rather than Polygons (the actual tarmac)
        # We need to handle both
        print("....aeroways")
        runways_polys = runways_rot[runways_rot.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        runways_lines = runways_rot[runways_rot.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        
        airway_color = THEME.get("aeroway", THEME["road_motorway"])  # see if aeroway color is defined, else fall back on road_motorway
        
        # Render Polygons (Runway surfaces)
        if not runways_polys.empty:
            #large_runways = rmkbl_polys[rmkbl_polys.length > 200]
            runways_polys.plot(ax=ax, 
                        facecolor=airway_color, 
                        edgecolor=airway_color,  # Set edge color to match to avoid 'background' bleed
                        linewidth=0.6, 
                        zorder=2)
        
        # Render Lines (Taxiways or Runway described as LineStrings)           
        if not runways_lines.empty:
            # Calculate Scale: How many meters per Matplotlib point?
            x_range = crop_xlim[1] - crop_xlim[0]
            fig_width_inches = fig.get_size_inches()[0]
            m_to_pt_ratio = (fig_width_inches / x_range) * 72
            
            # 4. Improved Width logic with visibility floor
            def calculate_linewidth(row):
                a_type = row.get('aeroway', 'runway')
                w = row.get('width')
                
                # Default physical widths in meters
                default_w = 45.0 if a_type == 'runway' else 15.0
                
                # Try to get physical width from tags
                try:
                    if w is None or (isinstance(w, float) and np.isnan(w)):
                        val = default_w
                    elif isinstance(w, str):
                        val = float(''.join(filter(lambda x: x.isdigit() or x == '.', w)))
                    else:
                        val = float(w)
                except (ValueError, TypeError):
                    val = default_w
                
                # Increase the visibility floor to 0.8
                return max(val * m_to_pt_ratio, 0.8)
            
            runways_lines.loc[:, 'lw_pt'] = runways_lines.apply(calculate_linewidth, axis=1)
            
            # 2. Plotting
            # Slightly higher in zorder than polygons to avoid flickering
            for _, row in runways_lines.iterrows():
                lw = row['lw_pt']
                geom = row.geometry
                a_type = row.get('aeroway', 'unknown')
                raw_width = row.get('width', 'N/A')
                
                #print(f"  -> Plotting {a_type}: RawWidth={raw_width}, CalcLW={lw:.4f} pts, Geom={geom.geom_type}")
                
                if geom.geom_type == 'LineString':
                    x, y = geom.xy
                    # solid_capstyle='butt' ensures the lines have flat ends, which looks much more pro for airport infrastructure than the default rounded ends
                    ax.plot(x, y, color=airway_color, linewidth=lw, solid_capstyle='butt', zorder=5.1)
                elif geom.geom_type == 'MultiLineString':
                    #print(f"     (Complex geometry with {len(geom.geoms)} parts)")
                    for line in geom.geoms:
                        x, y = line.xy
                        ax.plot(x, y, color=airway_color, linewidth=lw, solid_capstyle='butt', zorder=5.1)

    # Layer: railways (⚠ railways like roads are graphs)                                           
    if railways_rot is not None and len(railways_rot.edges) > 0:
        print("....railways")
        railway_color = THEME.get("railway", THEME["road_secondary"])  # see if railway color is defined, else fall back on road_secondary
        # Convert railways MultiDiGraph to GeoDataFrame
        _, railways_rot_gdf = ox.graph_to_gdfs(railways_rot)
        railways_rot_gdf.plot(ax=ax, 
                        color=railway_color, 
                        linewidth=edge_widths,  # Slightly thinner than roads to avoid overpowering the map
                        linestyle=(0, (5, 2)), # Dashed line for classic railway look
                        zorder=2.5)
    
    # Layer: Roads with hierarchy coloring
    print("....Roads (with hierarchy colors)")

    # Plot the projected graph
    # Convert road graph from OSMNX to GeoDataFrame for faster plotting
    #_, roads_rot_gdf = ox.graph_to_gdfs(g_rotated)
    #roads_rot_gdf.plot(ax=ax, color=edge_colors, linewidth=edge_widths, linestyle='solid', zorder=6)

    ox.plot_graph(
        g_rotated, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False
    )

    # # FORCE Matplotlib to flush the dashed cache from the railway layer
    # #ax.collections[-1] grabs the very last thing plotted (the roads)
    # if ax.collections:
    #     ax.collections[-1].set_linestyle('solid')
    #     ax.collections[-1].set_dashes([(0, None)]) # Completely clears the dash pattern

    # Apply the cropped limits
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    # Layer: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    ax.set_aspect("equal", adjustable="box")
    if show_north:
        draw_north_badge(ax, orientation_offset, THEME["text"])

    # -------------
    # 6. Text

    # Calculate scale factor based on smaller dimension (reference 12 inches)
    # This ensures text scales properly for both portrait and landscape orientations
    scale_factor = min(height, width) / 12.0

    # Base font sizes (at 12 inches width)
    base_main = 60
    base_sub = 22
    base_coords = 14
    base_attr = 8

    # Typography - use custom fonts if provided, otherwise use default FONTS
    active_fonts = fonts or FONTS
    if active_fonts:
        # font_main is calculated dynamically later based on length
        font_sub = FontProperties(
            fname=active_fonts["light"], size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            fname=active_fonts["regular"], size=base_coords * scale_factor
        )
        font_attr = FontProperties(
            fname=active_fonts["light"], size=base_attr * scale_factor
        )
    else:
        # Fallback to system fonts
        font_sub = FontProperties(
            family="monospace", weight="normal", size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            family="monospace", size=base_coords * scale_factor
        )
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    # Format city name based on script type
    # Latin scripts: apply uppercase and letter spacing for aesthetic
    # Non-Latin scripts (CJK, Thai, Arabic, etc.): no spacing, preserve case structure
    if is_latin_script(display_city):
        # Latin script: uppercase with letter spacing (e.g., "P  A  R  I  S")
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        # Non-Latin script: no spacing, no forced uppercase
        # For scripts like Arabic, Thai, Japanese, etc.
        spaced_city = display_city

    # Dynamically adjust font size based on city name length to prevent truncation
    # We use the already scaled "main" font size as the starting point.
    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)

    # Heuristic: If length is > 10, start reducing.
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(
            fname=active_fonts["bold"], size=adjusted_font_size
        )
    else:
        font_main_adjusted = FontProperties(
            family="monospace", weight="bold", size=adjusted_font_size
        )

    # --- BOTTOM TEXT ---
    ax.text(
        0.5,
        0.14,
        spaced_city,
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_main_adjusted,
        zorder=11,
    )

    ax.text(
        0.5,
        0.10,
        display_country.upper(),
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_sub,
        zorder=11,
    )

    lat, lon = point
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(
        0.5,
        0.07,
        coords,
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.7,
        ha="center",
        fontproperties=font_coords,
        zorder=11,
    )

    ax.plot(
        [0.4, 0.6],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=THEME["text"],
        linewidth=1 * scale_factor,
        zorder=11,
    )

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS["light"], size=8)
    else:
        font_attr = FontProperties(family="monospace", size=8)

    ax.text(
        0.98,
        0.02,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.5,
        ha="right",
        va="bottom",
        fontproperties=font_attr,
        zorder=11,
    )

    # -------------
    # 7. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(
        facecolor=THEME["bg"],
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✔ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid

  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout

  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula

  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # i18n support with custom display names
  python create_map_poster.py -c "Tokyo" -C "Japan" -dc "東京" -dC "日本" --font-family "Noto Sans JP" -t japanese_ink
  
  # example of rotation 90deg with north arrow
  python create_map_poster.py -c "SPA-Francorchamps" -C "Belgium" -t grand_prix -d 6000 --font-family "Russo One" -O -90
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c                  City name (required)
  --display-city, -dc         Custom display name for city (for i18n support)
  --country, -C               Country name (required)
  --display-country, -dC      Custom display name for country (for i18n support)
  --theme, -t                 Theme name (default: feature_based)
  --all-themes                Generate posters for all themes
  --distance, -d              Map radius in meters (default: 29000)
  --list-themes               List all available themes
  --width, -W                 Image width in inches (default: 12)
  --height, -H                Image height in inches (default: 16)
  --format, -f                Output format for the poster ('png', 'svg', 'pdf') (default: png)
  --fonts                     Google Fonts family name (e.g., "Noto Sans JP", "Open Sans"). If not specified, uses local Roboto fonts.
  --fast                      Fast mode: fetches only driving roads (faster but less detailed)
  --include-railways, -iR     Render railways
  --orientation-offset, -O    Rotation of the map (Allowed range: `-180` to `180`)
  --show-north                Show the north badge (even if orientation =0). If not set, will not show. Will automatially show if <> 0 but could be forced hidden with --no-show-north)
  --no-show-north             North badge is forced hidden
  
Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city "New York" --country "USA" -l 40.776676 -73.971321 --theme neon_cyberpunk
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """,
    )

    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument("--latitude",
        "-lat",
        dest="latitude",
        type=str,
        help="Override latitude center point",
    )
    parser.add_argument("--longitude",
        "-long",
        dest="longitude",
        type=str,
        help="Override longitude center point",
    )
    parser.add_argument("--theme",
        "-t",
        type=str,
        default="terracotta",
        help="Theme name (default: terracotta)",
    )
    parser.add_argument("--all-themes",
        "--All-themes",
        dest="all_themes",
        action="store_true",
        help="Generate posters for all themes",
    )
    parser.add_argument("--distance",
        "-d",
        type=int,
        default=18000,
        help="Map radius in meters (default: 18000)",
    )
    parser.add_argument("--width",
        "-W",
        type=float,
        default=12,
        help="Image width in inches (default: 12, max: 20 )",
    )
    parser.add_argument("--height",
        "-H",
        type=float,
        default=16,
        help="Image height in inches (default: 16, max: 20)",
    )
    parser.add_argument("--list-themes", action="store_true", help="List all available themes")
    parser.add_argument("--display-city",
        "-dc",
        type=str,
        help="Custom display name for city (for i18n support)",
    )
    parser.add_argument("--display-country",
        "-dC",
        type=str,
        help="Custom display name for country (for i18n support)",
    )
    parser.add_argument("--font-family",
        type=str,
        help='Google Fonts family name (e.g., "Noto Sans JP", "Open Sans"). If not specified, uses local Roboto fonts.',
    )
    parser.add_argument("--format",
        "-f",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Output format for the poster (default: png)",
    )
    parser.add_argument("--fast",
        dest="fast_mode",
        action="store_true",
        help="Fast mode: fetches only driving roads (faster but less detailed)",
    )
    parser.add_argument("--include-railways",
        "-iR",
        dest="include_railways",
        action="store_true",
        help="Enable railways rendering",
    )
    parser.set_defaults(include_railways=False)
    parser.add_argument("--orientation-offset",
        "-O",
        dest="orientation_offset",
        type=float,
        default=0.0,
        help="Map orientation offset in degrees relative to north (clockwise positive, range: -180 to 180)",
    )
    parser.add_argument("--show-north", 
        dest="show_north",
        action=argparse.BooleanOptionalAction,
        help="Show the north badge (even if orientation =0). If not set, will not show. Will automatially show if <> 0 but could be forced hidden with --no-show-north)",
    )

    args = parser.parse_args()

    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    # Enforce maximum dimensions
    if args.width > 20:
        print(
            f"⚠ Width {args.width} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.width = 20.0
    if args.height > 20:
        print(
            f"⚠ Height {args.height} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.height = 20.0
    if not -180 <= args.orientation_offset <= 180:
        print(
            f"Error: --orientation-offset must be between -180 and 180. Received {args.orientation_offset}."
        )
        sys.exit(1)

    if args.show_north: # if set then show badge
        show_north = True
    elif args.show_north is None: # if not set, then show badge only if orientation <> 0
        show_north = args.orientation_offset != 0
        print(f"show_north not defined, set to {show_north} based on orientation_offset {args.orientation_offset}")
    else:  # if arg is --no-show-north, then  args.show_north is false.
        show_north = False

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"❌ Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Load custom fonts if specified
    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    # Get coordinates and generate poster
    try:
        if args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = [lat, lon]
            print(f"✔ Coordinates: {', '.join([str(i) for i in coords])}")
        else:
            coords = get_coordinates(args.city, args.country)

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(
                args.city,
                args.country,
                coords,
                args.distance,
                output_file,
                args.format,
                args.width,
                args.height,
                display_city=args.display_city,
                display_country=args.display_country,
                fonts=custom_fonts,
                fast_mode=args.fast_mode,
                include_railways=args.include_railways,
                orientation_offset=args.orientation_offset,
                show_north=show_north,
            )

        print("\n" + "=" * 50)
        print("✔ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
