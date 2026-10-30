"""Build tool/index.html: inline engine.js and the 10-minute feeder data.
usage (from the repo root, after run_all.py):  python tool/build.py"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import weather  # noqa: E402

f = pd.read_csv(os.path.join(HERE, "..", "data", "derived", "feeder_1min.csv.gz"))
b10 = f["load_kw"].to_numpy().reshape(-1, 10).mean(axis=1)
w = weather.load()
data = {"start": "2018-06-01", "base10": [int(round(x * 10)) for x in b10],
        "temp": [int(round(x * 10)) for x in w["t_air"]]}
html = open(os.path.join(HERE, "template.html")).read()
html = html.replace("/*__ENGINE__*/", open(os.path.join(HERE, "engine.js")).read())
html = html.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
# artifact.html: body content only (the hosting page adds the html/head/body skeleton)
open(os.path.join(HERE, "artifact.html"), "w").write(html)
# index.html: full page for GitHub Pages or opening locally
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
open(os.path.join(HERE, "index.html"), "w").write(page)
print("wrote artifact.html and index.html", len(page) // 1024, "KB")
