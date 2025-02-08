import geopandas as gpd
import osmnx as ox

def main():
    # city = "porto"
    city = "beijing"
    edge_shp_file_path = "./{}/{}_shp/edges.shp".format(city, city)
    node_shp_file_path = "./{}/{}_shp/nodes.shp".format(city, city)
    edges_gdf = gpd.read_file(edge_shp_file_path)
    nodes_gdf = gpd.read_file(node_shp_file_path)

    edge_uv_dict = {}
    for index, row in edges_gdf.iterrows():
        fid = row["fid"]
        start_p = row["u"]  # start point
        end_p = row["v"]  # end point
        edge_uv_dict[fid] = (start_p, end_p)
    
    adj_dict = {}
    for edge_1 in edge_uv_dict:
        for edge_2 in edge_uv_dict:
            if edge_1 != edge_2:
                end_p1 = edge_uv_dict[edge_1][1]
                start_p2 = edge_uv_dict[edge_2][0]
                if end_p1 == start_p2:
                    if edge_1 not in adj_dict:
                        adj_dict[edge_1] = [edge_2]
                    else:
                        adj_dict[edge_1].append(edge_2)

    adj_list_file = open("./{}/raw_data/roadnet.adjlist".format(city), "w")
    for edge in adj_dict:
        adjs = adj_dict[edge]
        line = str(edge)
        for adj in adjs:
            line = line + " " + str(adj)
        line += "\n"
        adj_list_file.write(line)
                

main()
