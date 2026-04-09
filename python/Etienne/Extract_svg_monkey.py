from itertools import cycle
from os import name

import h5py
from svgpathtools import svg2paths
import numpy as np

paths, attributes = svg2paths("D:\\Carlen\\Nouveau dossier\\flatmap_macaque_extract.svg")

import re

def parse_matrix(transform_str):
    nums = list(map(float, re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", transform_str)))
    a, b, c, d, e, f = nums
    return np.array([[a, c, e],
                     [b, d, f],
                     [0, 0, 1]])

def apply_transform(points, M):
    pts = np.c_[points, np.ones(len(points))]
    transformed = pts @ M.T
    return transformed[:, :2]

def sample_path(path, n=500):
    ts = np.linspace(0, 1, n)
    pts = np.array([path.point(t) for t in ts])
    return np.column_stack((pts.real, pts.imag))

def is_real_shape(attr):
    style = attr.get("style", "")
    if not style:
        return False  # clip paths, defs, masks
    '''
    if "clip-path" in attr:
        return False
    '''
    if "fill:none" in style and "stroke:none" in style:
        return False
    
    return True

from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET

import re

def extract_font_size(style):
    m = re.search(r"font-size:([0-9.]+)px", style)
    return float(m.group(1)) if m else 12.0

def estimate_text_width(label, font_size):
    return 0.6 * font_size * len(label)
def text_bbox(label, font_size):
    w = estimate_text_width(label, font_size)
    h = font_size  # approximate height
    return np.array([
        [0, 0],
        [w, 0],
        [w, -h],
        [0, -h]
    ])


def extract_texts(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = {"svg": "http://www.w3.org/2000/svg"}

    texts = []

    for t in root.findall(".//svg:text", ns):
        label = "".join(t.itertext()).strip()

        # Default position
        x = float(t.get("x", 0))
        y = float(t.get("y", 0))

        # Apply transform if present
        if "transform" in t.attrib:
            M = parse_matrix(t.attrib["transform"])
            px, py, _ = M @ np.array([x, y, 1])
        else:
            px, py = x, y
        
        font_size = extract_font_size(t.get("style", ""))
        bbox = text_bbox(label, font_size)
        bbox_transformed = apply_transform(bbox, M)

        cx = bbox_transformed[:, 0].mean()
        cy = bbox_transformed[:, 1].mean()

        texts.append({
            "label": label,
            "x": px,
            "y": py,
            "cx": cx,
            "cy": cy
        })

    return texts




def find_region_for_text(text, regions):
    p = Point(text["x"], text["y"])

    for region in regions:
        poly = Polygon(region["points"])
        if poly.contains(p):
            return region

    return None


regions = []

filtered = [
    (path, attr)
    for path, attr in zip(paths, attributes)
    if is_real_shape(attr)
]

for path, attr in filtered:
    

    pts = sample_path(path, n=800)

    if "transform" in attr:
        M = parse_matrix(attr["transform"])
        pts = apply_transform(pts, M)

    fill = re.search('(?<=fill:)#\\w+', attr['style'])
    if fill:
        fill = fill.group(0)
    regions.append({"points": pts, "fill": fill, "id": attr.get("id"), "style": attr.get("style")})

texts = extract_texts("D:\\Carlen\\Nouveau dossier\\flatmap_macaque_extract.svg")




links = []

for t in texts:
    region = find_region_for_text(t, regions)
    region["label"] = t["label"]
    links.append({
        "label": t["label"],
        "x": t["x"],
        "y": t["y"],
        "region_id": region["id"] if region else None,
        "region_style": region["style"] if region else None
    })

from shapely.geometry import Polygon

outline, area = None, 0
for r,region in enumerate(regions):
    poly = Polygon(region["points"])
    region["centroid"] = np.array(poly.centroid.coords[0])
    if poly.area > area:
        area = poly.area
        outline = r


for link in links:
    if link["region_id"] is not None:
        region = next(r for r in regions if r["id"] == link["region_id"])
        link["cx"], link["cy"] = region["centroid"]
    else:
        # fallback: keep original text position
        link["cx"], link["cy"] = link["x"], link["y"]

flatmap_dir = "D:\\Carlen\\Intermediate\\flatmaps\\flatmap_monkey"

with h5py.File(f"{flatmap_dir}/flatmap_regions.h5", "w") as f:
    for r, region in enumerate(regions):
        if r == outline:
            continue
        elif "label" not in region:
            f.create_dataset(f"unlabeled_{r}", data=region["points"])
        else:
            f.create_dataset(region["label"], data=region["points"])
            #Add the fill argument as an attribute to the dataset
            f[region["label"]].attrs['fill'] = region['fill']

#Save the outline into a txt file
with open(f"{flatmap_dir}/outline.txt", "w") as f:
    for point in regions[outline]["points"]:
        f.write(f"{point[0]},{point[1]}\n")


import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"  # keep text as text in SVG output

plt.figure(figsize=(8, 10))

for r, region in enumerate(regions):
    if r == outline:
        plt.fill(region["points"][:,0], region["points"][:,1], color='none', edgecolor='red', linewidth=5)
        continue
    plt.fill(region["points"][:,0], region["points"][:,1], color=region['fill'], alpha=0.5, edgecolor='black', linewidth=2)
for link in links:
    plt.text(
        link["cx"], link["cy"],
        link["label"],
        fontsize=8,
        ha="center", va="center",
        color="black"
    )


plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()  # SVG coordinate system
plt.show()
