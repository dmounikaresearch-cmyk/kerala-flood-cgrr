"""Study-area map (manuscript Fig. 2). Run in notebook 02 after the experiments:
   exec(open('make_study_area_map.py').read())   # needs DATA = path to Paper4_Kerala
"""
import geopandas as gpd, rasterio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, base64, io
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
D = DATA
dist = gpd.read_file(D + '/kerala_districts.geojson').to_crs(4326)
roads = gpd.read_file(D + '/kerala_roads.gpkg').to_crs(4326)
ls = gpd.read_file(D + '/landslides_2018/landslides_2018.shp').to_crs(4326)
l19 = pd.read_csv(D + '/landslides_2019.csv')
from rasterio.features import geometry_mask
from affine import Affine
with rasterio.open(D + '/floodmask_20180821.tif') as r:
    a = r.read(1, out_shape=(r.height // 2, r.width // 2)); b = r.bounds
    tr = r.transform * Affine.scale(r.width / a.shape[1], r.height / a.shape[0])
land = geometry_mask(list(dist.geometry), out_shape=a.shape, transform=tr, invert=True)
a = np.where(land, a, 0)
hubs = {'Thrissur': (10.527, 76.214), 'Ernakulam': (9.981, 76.299), 'Thiruvananthapuram': (8.524, 76.936), 'Kozhikode': (11.258, 75.780)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
fig, ax = plt.subplots(figsize=(4.4, 6.0))
roads.plot(ax=ax, color='0.6', linewidth=0.15, zorder=1)
fm = np.ma.masked_where(a != 1, a)
ax.imshow(fm, extent=(b.left, b.right, b.bottom, b.top), cmap=matplotlib.colors.ListedColormap(['#1f77b4']), zorder=2, interpolation='nearest')
dist.boundary.plot(ax=ax, color='k', linewidth=0.5, zorder=3)
ax.scatter(ls.geometry.x, ls.geometry.y, s=0.6, c='#d95f02', linewidths=0, zorder=4)
ax.scatter(l19.lon, l19.lat, s=28, marker='^', c='#e7298a', edgecolors='k', linewidths=0.4, zorder=6)
for n, (la, lo) in hubs.items():
    ax.scatter(lo, la, s=55, marker='*', c='gold', edgecolors='k', linewidths=0.5, zorder=7)
    ax.annotate(n, (lo, la), xytext=(4, -6 if n != 'Kozhikode' else 4), textcoords='offset points', fontsize=6.5, zorder=8)
for _, rw in dist.iterrows():
    p = rw.geometry.representative_point(); ax.text(p.x, p.y, rw['district'], fontsize=5, color='0.25', ha='center', zorder=5)
ax.set_xlim(74.8, 77.5); ax.set_ylim(8.2, 12.85); ax.set_aspect('equal')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.legend(handles=[Line2D([], [], color='0.6', lw=0.8, label='Main roads (OSM)'), Patch(color='#1f77b4', label='Flood extent, 21 Aug 2018'),
                   Line2D([], [], marker='o', ls='', ms=2, color='#d95f02', label='Landslides 2018 (Hao et al. 2020)'),
                   Line2D([], [], marker='^', ls='', ms=5, mfc='#e7298a', mec='k', label='Landslides 2019'),
                   Line2D([], [], marker='*', ls='', ms=7, mfc='gold', mec='k', label='Staging hubs')], loc='lower left', fontsize=5.5, frameon=True)
fig.tight_layout(); fig.savefig(D + '/results/figures/Fig_study_area.png', dpi=400)
buf = io.BytesIO(); fig.savefig(buf, format='jpg', dpi=300, pil_kwargs={'quality': 90})
MB = base64.b64encode(buf.getvalue()).decode(); CK2 = 0; print('LEN', len(MB))
