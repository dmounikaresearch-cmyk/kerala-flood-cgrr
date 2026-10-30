"""Build realized route travel times per date from the road graph,
flooded road segments (notebook 01) and landslide road blockages.

road mode : flooded and landslide-blocked segments are impassable
boat mode : landslide-blocked segments impassable; flooded segments
            passable at travel_time x boat_slowdown (boats on water)
"""
import numpy as np
import pandas as pd
import networkx as nx


def blocked_edges_from_landslides(edges_gdf, landslides_gdf, buffer_m=50):
    """edge_ids within buffer_m of any landslide point (both in any CRS)."""
    import geopandas as gpd
    e = edges_gdf.to_crs(32643)
    l = landslides_gdf.to_crs(32643)
    j = gpd.sjoin_nearest(l, e[["edge_id", "geometry"]], max_distance=buffer_m, how="inner")
    return set(j["edge_id"].astype(int))


def route_times(G, edge_table, origins, dests, dates, flooded_by_date, blocked_by_date,
                boat_slowdown=3.0, weight="travel_time"):
    """
    G               : networkx (Multi)DiGraph with `weight` in seconds
    edge_table      : DataFrame edge_id, u, v, key
    origins, dests  : dict name -> node
    dates           : list of date strings
    flooded_by_date : dict date -> set(edge_id)
    blocked_by_date : dict date -> set(edge_id)  (landslides active on that date)
    returns DataFrame date, origin, district, mode, tt_hours (inf = impassable)
    """
    et = edge_table.set_index("edge_id")
    Gu = G.to_undirected()
    base_w = {}
    for u, v, k, d in Gu.edges(keys=True, data=True):
        base_w[(u, v, k)] = float(d.get(weight, 1.0))

    def ekeys(ids):
        out = set()
        for e in ids:
            if e in et.index:
                r = et.loc[e]
                out.add((r.u, r.v, r.key))
                out.add((r.v, r.u, r.key))
        return out

    rows = []
    for date in dates:
        fl = ekeys(flooded_by_date.get(date, set()))
        bl = ekeys(blocked_by_date.get(date, set()))
        for mode in ("road", "boat"):
            H = nx.MultiGraph()
            H.add_nodes_from(Gu.nodes)
            for (u, v, k), w in base_w.items():
                if (u, v, k) in bl:
                    continue
                if (u, v, k) in fl:
                    if mode == "road":
                        continue
                    w = w * boat_slowdown
                H.add_edge(u, v, key=k, w=w)
            for o, on in origins.items():
                dist = nx.single_source_dijkstra_path_length(H, on, weight="w")
                for dname, dn in dests.items():
                    tt = dist.get(dn, np.inf)
                    rows.append({"date": date, "origin": o, "district": dname,
                                 "mode": mode, "tt_hours": tt / 3600.0})
    return pd.DataFrame(rows)
