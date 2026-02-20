"""
Rotation Management adapted as a Module from https://github.com/juandesant/maptoposter/
"""

import argparse
import colorsys
import numpy as np
import osmnx as ox
import networkx as nx
from pathlib import Path
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
from geopandas import GeoDataFrame
from matplotlib.patches import Polygon
from shapely import affinity
from shapely.geometry import Point


COMPASS_SVG_PATH = Path("compass.svg")
_COMPASS_SHAPES_CACHE = None


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


def parse_bool_arg(value: str) -> bool:
    """
    Parse CLI boolean values from common true/false strings.
    """
    normalized = value.strip().lower()
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false."
    )


def _parse_svg_points(points_str):
    """
    Parse SVG polygon point lists into numeric (x, y) tuples.
    """
    coords = []
    for token in points_str.replace("\n", " ").replace("\t", " ").split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        coords.append((float(x_str), float(y_str)))
    return coords


def _load_compass_shapes():
    """
    Load polygon geometry and style from compass.svg.
    Returns a list of dicts: {"points": [...], "fill": str, "stroke": str}
    """
    global _COMPASS_SHAPES_CACHE
    if _COMPASS_SHAPES_CACHE is not None:
        return _COMPASS_SHAPES_CACHE

    if not COMPASS_SVG_PATH.exists():
        _COMPASS_SHAPES_CACHE = []
        return _COMPASS_SHAPES_CACHE

    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.parse(COMPASS_SVG_PATH).getroot()
    symbol = root.find(".//svg:symbol[@id='elm']", ns)
    if symbol is None:
        _COMPASS_SHAPES_CACHE = []
        return _COMPASS_SHAPES_CACHE

    shapes = []
    for poly in symbol.findall("svg:polygon", ns):
        points = _parse_svg_points(poly.attrib.get("points", ""))
        if not points:
            continue
        shapes.append(
            {
                "points": points,
                "fill": poly.attrib.get("fill", "#FBFBFB"),
                "stroke": poly.attrib.get("stroke", "#333"),
            }
        )

    _COMPASS_SHAPES_CACHE = shapes
    return _COMPASS_SHAPES_CACHE


def _as_hex(color):
    """
    Convert a Matplotlib color string to canonical hex.
    """
    return mcolors.to_hex(color).lower()


def _gray_to_main_shade(source_color, main_color):
    """
    Map grayscale source colors to tonal shades of the main theme color.
    """
    src_r, src_g, src_b = mcolors.to_rgb(source_color)
    luminance = (src_r + src_g + src_b) / 3.0

    main_r, main_g, main_b = mcolors.to_rgb(main_color)
    hue, _lightness, saturation = colorsys.rgb_to_hls(main_r, main_g, main_b)

    # Keep hue/saturation and vary lightness from dark to light tones.
    target_lightness = 0.18 + (0.68 * luminance)
    out_r, out_g, out_b = colorsys.hls_to_rgb(hue, target_lightness, saturation)
    return mcolors.to_hex((out_r, out_g, out_b))


def draw_north_badge(ax, orientation_offset, main_color):
    """
    Draw a north orientation badge in projected map coordinates.
    """
    # Anchor in map-local data coordinates (projected x/y, not lat/lon).
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    cx = xlim[0] + (x_range * 0.08)
    cy = ylim[0] + (y_range * 0.90)

    compass_shapes = _load_compass_shapes()
    theta = np.deg2rad(orientation_offset)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    gray_palette = {"#ffffff", "#fbfbfb", "#999999", "#333333", "#000000"}

    # Max diameter = 5% of image/map size -> radius = 2.5% of min extent.
    max_compass_radius = 0.025 * min(x_range, y_range)
    max_source_radius = 0.0
    for shape in compass_shapes:
        for x, y in shape["points"]:
            max_source_radius = max(max_source_radius, (x * x + y * y) ** 0.5)
    compass_scale = (
        max_compass_radius / max_source_radius if max_source_radius > 0 else 0
    )
    compass_scale_x = compass_scale
    compass_scale_y = compass_scale

    # Positive user offset is clockwise; SVG point rotation math below is clockwise.
    for shape in compass_shapes:
        fill_hex = _as_hex(shape["fill"])
        stroke_hex = _as_hex(shape["stroke"])
        fill_color = (
            _gray_to_main_shade(fill_hex, main_color)
            if fill_hex in gray_palette
            else shape["fill"]
        )
        stroke_color = (
            main_color if stroke_hex in gray_palette else shape["stroke"]
        )

        transformed_points = []
        for x, y in shape["points"]:
            rx = x * cos_t + y * sin_t
            ry = -x * sin_t + y * cos_t
            transformed_points.append(
                (cx + (rx * compass_scale_x), cy + (ry * compass_scale_y))
            )

        ax.add_patch(
            Polygon(
                transformed_points,
                closed=True,
                facecolor=fill_color,
                edgecolor=stroke_color,
                linewidth=0.8,
                zorder=13,
            )
        )

    # Fallback if compass.svg is unavailable or malformed.
    if not compass_shapes:
        arrow_len = max_compass_radius
        dx = np.sin(theta) * arrow_len
        dy = np.cos(theta) * arrow_len
        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),
            xytext=(cx, cy),
            arrowprops={
                "arrowstyle": "-|>",
                "color": main_color,
                "linewidth": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=13,
        )

    label_dir_x = np.sin(theta)
    label_dir_y = np.cos(theta)
    label_norm = np.hypot(label_dir_x, label_dir_y)
    if label_norm == 0:
        label_norm = 1.0
    unit_x = label_dir_x / label_norm
    unit_y = label_dir_y / label_norm

    # Place label 12 screen pixels away from the compass north tip.
    tip_x = cx + unit_x * max_compass_radius
    tip_y = cy + unit_y * max_compass_radius
    tip_disp = ax.transData.transform((tip_x, tip_y))
    center_disp = ax.transData.transform((cx, cy))
    disp_dir = tip_disp - center_disp
    disp_norm = np.hypot(disp_dir[0], disp_dir[1])
    if disp_norm == 0:
        disp_dir = np.array([0.0, 1.0])
        disp_norm = 1.0
    label_disp = tip_disp + (disp_dir / disp_norm) * 12.0
    label_x, label_y = ax.transData.inverted().transform(label_disp)

    ax.text(
        label_x,
        label_y,
        "N",
        color=main_color,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=14,
    )
    

def get_projected_center(g_proj, center_lat_lon):
    """
    Project center point into graph CRS and return metric x/y coordinates.
    """
    lat, lon = center_lat_lon
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"],
        )[0]
    )
    return center.x, center.y


def rotate_graph_and_features(g_proj, features, center_lat_lon, angle):
    if angle == 0:
        return g_proj, features
        
    # Get center in projected coordinates
    center_x, center_y = get_projected_center(g_proj, center_lat_lon)
    angle_ccw = -angle  # Convert clockwise offset to counter-clockwise for Shapely
    
    # 1. Rotate Graph Nodes
    g_rotated = g_proj.copy()
    for node, data in g_rotated.nodes(data=True):
        p = Point(data['x'], data['y'])
        p_rot = affinity.rotate(p, angle_ccw, origin=(center_x, center_y))
        data['x'], data['y'] = p_rot.x, p_rot.y
    
    # 2. IMPORTANT: ROTATE EVERY EDGE GEOMETRY (Crucial for roads/railways)
    # If we only rotate nodes, OSMnx's plotting functions will still draw the original unrotated 
    # road shapes between the new node positions.
    for u, v, k, data in g_rotated.edges(data=True, keys=True):
        if 'geometry' in data:
            data['geometry'] = affinity.rotate(
                data['geometry'], 
                angle_ccw, 
                origin=(center_x, center_y)
            )

    # Rotate GeoDataFrames
    rotated_features = []
    for item in features:
        # CASE A: Handle None or empty items early
        if item is None:
            rotated_features.append(None)
            continue

       # CASE B: Handle GeoDataFrames (Parks, Water, etc.)
        if isinstance(item, GeoDataFrame):
            if item.empty:
                rotated_features.append(item)
                continue
            
            gdf_rot = item.copy()
            gdf_rot["geometry"] = gdf_rot.geometry.apply(
                lambda geom: affinity.rotate(
                    geom, angle_ccw, origin=(center_x, center_y)
                )
            )
            rotated_features.append(gdf_rot)

        # CASE C: Handle MultiDiGraphs (Railways)
        elif isinstance(item, (nx.MultiDiGraph, nx.Graph)):
            if not item: # Check if graph is empty (0 nodes)
                rotated_features.append(item)
                continue
                
            g_item_rot = item.copy()
            # Rotate nodes
            for _, data in g_item_rot.nodes(data=True):
                p = Point(data['x'], data['y'])
                p_rot = affinity.rotate(p, angle_ccw, origin=(center_x, center_y))
                data['x'], data['y'] = p_rot.x, p_rot.y
            
            # Rotate edge geometries
            for _, _, _, data in g_item_rot.edges(data=True, keys=True):
                if 'geometry' in data:
                    data['geometry'] = affinity.rotate(
                        data['geometry'], 
                        angle_ccw, 
                        origin=(center_x, center_y)
                    )
            rotated_features.append(g_item_rot)
            
        else:
            # Unknown type, just pass it through
            rotated_features.append(item)

    return g_rotated, rotated_features