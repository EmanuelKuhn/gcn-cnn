# Steps:

1. Get FP_box_info.pkl from: https://github.com/agp-ka32/LayoutGMN-pytorch#preparing-layout-graphs and store in layoutmgn_data
2. Run `python preprocessing/0_cal_geometry_feat.py`
3. Run `python preprocessing/1_build_geometry_graph.py`
4. Run `python preprocessing/compute_25Chan_Imgs.py`

Use `the create_custom_apn_dict.ipynb` notebook to make training triplets using your own IoU labels, or with different splits.