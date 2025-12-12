import os
import json
import argparse
import pandas as pd

# Scene lists
M360_SCENES = [
    "bicycle", "flowers", "garden", "stump", "treehill",
    "room", "counter", "kitchen", "bonsai"
]
NERF_SYN_SCENES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship"
]
EXTEND_SCENES = [
    "drjohnson", "playroom", "train", "truck"
]                               

# Example methods (adjust to your actual folder names or however you want to label them):
METHODS = [
    "FR_display_0.99"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_path",
        type=str,
        required=True,
        help="Root path containing subfolders for each method."
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="results.xlsx",
        help="Path to output Excel file."
    )
    return parser.parse_args()


def read_json(filepath):
    """Safely read JSON file into Python dict."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except:
            return None


def collect_scene_metrics(scene_path):
    """
    Returns a dict of the relevant metrics from:
      - results.json
      - ours_30000_rendering_powers.json
    If either file is missing or invalid, returns None (or partial data).
    """
    results_json_path = os.path.join(scene_path, "levels/L1/results.json")
    rendering_json_path = os.path.join(scene_path, "levels/FR_rendering_powers.json")
    display_results_json_path = os.path.join(scene_path, "levels/results.json")
    
    results_data = read_json(results_json_path)
    rendering_data = read_json(rendering_json_path)
    display_results_data = read_json(display_results_json_path)

    # import ipdb; ipdb.set_trace()
    
    if not results_data or not rendering_data:
        return None
    
    top_key = list(results_data.keys())[0]
    psnr = results_data[top_key].get("PSNR", None)
    ssim = results_data[top_key].get("SSIM", None)
    lpips = results_data[top_key].get("LPIPS", None)
    # display_power = results_data[top_key].get("Display Power", None)
    
    # Extract from ours_30000_rendering_powers.json
    mean_metrics = rendering_data.get("mean_metrics", {})
    render_power = mean_metrics.get("mean_total_power", None)
    dram_power = mean_metrics.get("mean_total_dram_power", None)
    sram_power = mean_metrics.get("mean_total_sram_power", None)
    flops_power = mean_metrics.get("mean_total_flops_power", None)


    display_power = display_results_data[top_key].get("Display Power", None)
    display_power_new = display_results_data[top_key].get("Display Power New", None)
    
    # "Total Power" (for Excel) = "Render Power" + "Display Power"
    # as per your requirement that the JSON's "render power" is separate from "display power".
    total_power = render_power + display_power_new
    
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
        "Display Power": display_power,
        "Display Power New": display_power_new,
        "Render DRAM Power": dram_power,
        "Render SRAM Power": sram_power,
        "Render FLOPS Power": flops_power,
        "Render Power": render_power,
        "Total Power": total_power
    }


def gather_data_for_method(method_root, scenes):
    """
    method_root = /some/path/Min-Splatting-D/
    scenes = list of scene names, e.g. ["bicycle", "flowers", ...]
    Return a dict: { scene_name: { 'PSNR': x, 'SSIM': y, ... }, ... }
    """
    method_data = {}
    for scene in scenes:
        scene_path = os.path.join(method_root, scene)
        metrics = collect_scene_metrics(scene_path)
        method_data[scene] = metrics
    return method_data


def main():
    args = parse_args()
    method_path = args.method_path
    output_excel = args.output_excel

    
    all_data = {}
    
    # We only want to process subdirs that match METHODS (or if you want to auto-detect, you can).
    for method in METHODS:
        method_dir = os.path.join(method_path, method)
        if not os.path.isdir(method_dir):
            print(f"Warning: Method folder not found: {method_dir}")
            continue
        
        # Gather m360
        m360_dir = os.path.join(method_dir, "m360")
        if os.path.isdir(m360_dir):
            data_m360 = gather_data_for_method(m360_dir, M360_SCENES)
        else:
            data_m360 = {scene: None for scene in M360_SCENES}
        
        # Gather nerf_synthetic
        nerf_dir = os.path.join(method_dir, "nerf_synthetic")
        if os.path.isdir(nerf_dir):
            data_nerf = gather_data_for_method(nerf_dir, NERF_SYN_SCENES)
            # import ipdb; ipdb.set_trace()
        else:
            data_nerf = {scene: None for scene in NERF_SYN_SCENES}
        
        # Gather extend scenes
        extend_dir = os.path.join(method_dir, "extend")
        if os.path.isdir(extend_dir):
            data_extend = gather_data_for_method(extend_dir, EXTEND_SCENES)
        else:
            data_extend = {scene: None for scene in EXTEND_SCENES}
        
        all_data[method] = {
            "m360": data_m360,
            "nerf_synthetic": data_nerf,
            "extend": data_extend
        }
    
    desired_metrics_order = [
        "PSNR",
        "SSIM",
        "LPIPS",
        "Display Power",
        "Display Power New",
        "Render DRAM Power",
        "Render SRAM Power",
        "Render FLOPS Power",
        "Render Power",
        "Total Power"
    ]
    
    # Helper to create a multi-index for the rows:
    # We want something like:
    # Index = [
    #   (method="Min-Splatting-D", metric="PSNR"),
    #   (method="Min-Splatting-D", metric="SSIM"),
    #   ...
    #   (method="3DGS",           metric="PSNR"),
    #   ...
    # ]
    row_tuples_m360 = []
    row_tuples_nerf = []
    row_tuples_extend = []
    for method in METHODS:
        for metric in desired_metrics_order:
            row_tuples_m360.append((method, metric))
            row_tuples_nerf.append((method, metric))
            row_tuples_extend.append((method, metric))
    
    row_index_m360 = pd.MultiIndex.from_tuples(row_tuples_m360, names=["Method", "Metric"])
    row_index_nerf = pd.MultiIndex.from_tuples(row_tuples_nerf, names=["Method", "Metric"])
    row_index_extend = pd.MultiIndex.from_tuples(row_tuples_extend, names=["Method", "Metric"])
    
    # Columns = scene names
    df_m360 = pd.DataFrame(index=row_index_m360, columns=M360_SCENES)
    df_nerf = pd.DataFrame(index=row_index_nerf, columns=NERF_SYN_SCENES)
    df_extend = pd.DataFrame(index=row_index_extend, columns=EXTEND_SCENES)
    
    # Fill DataFrame for each method, each scene, each metric
    for method in METHODS:
        # m360
        for scene in M360_SCENES:
            metrics = all_data.get(method, {}).get("m360", {}).get(scene, None)
            if metrics is not None:
                for metric_key in desired_metrics_order:
                    df_m360.loc[(method, metric_key), scene] = metrics.get(metric_key, None)
        
        # nerf
        for scene in NERF_SYN_SCENES:
            metrics = all_data.get(method, {}).get("nerf_synthetic", {}).get(scene, None)
            if metrics is not None:
                for metric_key in desired_metrics_order:
                    df_nerf.loc[(method, metric_key), scene] = metrics.get(metric_key, None)
        
        # extend
        for scene in EXTEND_SCENES:
            metrics = all_data.get(method, {}).get("extend", {}).get(scene, None)
            if metrics is not None:
                for metric_key in desired_metrics_order:
                    df_extend.loc[(method, metric_key), scene] = metrics.get(metric_key, None)
    
    #-----------------------------------------------------------------------------------
    # 3) Write to Excel. 
    #    You can either:
    #      - Put m360 in one sheet, nerf_synthetic in another
    #      - Or place them side by side in the same sheet (more complicated layout).
    #
    #    Below we'll just create two sheets named 'm360' and 'nerf_synthetic'.
    #-----------------------------------------------------------------------------------
    
    with pd.ExcelWriter(output_excel) as writer:
        df_m360.to_excel(writer, sheet_name="m360")
        df_nerf.to_excel(writer, sheet_name="nerf_synthetic")
        df_extend.to_excel(writer, sheet_name="extend")
    
    print(f"Done! Results saved to {output_excel}")


if __name__ == "__main__":
    main()