import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from fit_a_function import fit_michaelis_menten_with_options

import sympy
import pickle

def parse_log_file(log_file):
    """
    Parse log.txt to retrieve initial and target metrics.
    
    For L1 (SSIM/PSNR):
      - "Initial SSIM: 0.7978"
      - "Initial PSNR: 25.5302"
      - "Target SSIM: 0.7898"
      - "Target PSNR: 25.2749"

    For L2, L3, L4 (HVS):
      - "Initial HVS: 1.3323862731340341e-05"
      - "Target HVS: 2.1341842511901632e-05"
    """
    metrics = {
        "initial_ssim": None,
        "initial_psnr": None,
        "target_ssim": None,
        "target_psnr": None,
        "initial_hvs": None,
        "target_hvs": None,
    }

    with open(log_file, "r") as f:
        log_text = f.read()

    # Regex patterns (L1)
    pattern_initial_ssim = r"Initial SSIM:\s*([\d\.]+)"
    pattern_initial_psnr = r"Initial PSNR:\s*([\d\.]+)"
    pattern_target_ssim  = r"Target SSIM:\s*([\d\.]+)"
    pattern_target_psnr  = r"Target PSNR:\s*([\d\.]+)"

    # Regex patterns (L2, L3, L4)
    pattern_initial_hvs = r"Initial HVS:\s*([\de\.\-+]+)"
    pattern_target_hvs  = r"Target HVS:\s*([\de\.\-+]+)"

    # Search in the log content
    m_init_ssim = re.search(pattern_initial_ssim, log_text)
    m_init_psnr = re.search(pattern_initial_psnr, log_text)
    m_tgt_ssim  = re.search(pattern_target_ssim, log_text)
    m_tgt_psnr  = re.search(pattern_target_psnr, log_text)
    m_init_hvs  = re.search(pattern_initial_hvs, log_text)
    m_tgt_hvs   = re.search(pattern_target_hvs, log_text)

    if m_init_ssim:
        metrics["initial_ssim"] = float(m_init_ssim.group(1))
    if m_init_psnr:
        metrics["initial_psnr"] = float(m_init_psnr.group(1))
    if m_tgt_ssim:
        metrics["target_ssim"] = float(m_tgt_ssim.group(1))
    if m_tgt_psnr:
        metrics["target_psnr"] = float(m_tgt_psnr.group(1))
    if m_init_hvs:
        metrics["initial_hvs"] = float(m_init_hvs.group(1))
    if m_tgt_hvs:
        metrics["target_hvs"] = float(m_tgt_hvs.group(1))

    return metrics


def load_json_results(json_file):
    """
    Load the specified JSON file if exists, else return None.
    """
    if not os.path.exists(json_file):
        return None
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def filter_data_L1(data_dict, target_ssim, target_psnr):
    """
    For L1: keep only entries where
      ssims[i] >= target_ssim * 0.99  AND  psnrs[i] >= target_psnr * 0.99
    Includes 1% tolerance below target values.
    """
    if data_dict is None:
        return None
    pruned_indices = []
    for i, (ssim_val, psnr_val) in enumerate(zip(data_dict["ssims"], data_dict["psnrs"])):
        if ssim_val >= target_ssim * 0.99 and psnr_val >= target_psnr * 0.99:
            pruned_indices.append(i)

    # Filter all lists by pruned_indices
    filtered_data = {}
    for key, vals in data_dict.items():
        filtered_data[key] = [vals[i] for i in pruned_indices]

    return filtered_data

def filter_data_Li(data_dict, target_hvs):
    """
    For L2, L3, L4: keep only entries where hvss[i] <=  target_hvs * 1.01
    Includes 1% tolerance above target value.
    """
    if data_dict is None:
        return None
    pruned_indices = []
    for i, hvs_val in enumerate(data_dict["hvss"]):
        if hvs_val <= target_hvs * 1.01:
            pruned_indices.append(i)

    filtered_data = {}
    for key, vals in data_dict.items():
        filtered_data[key] = [vals[i] for i in pruned_indices]

    return filtered_data




def find_solution(filename="/workspace/PowerGS/FR_Ours/my_symbolic_solution.txt", param_dict=None):
    """
    Reads a symbolic solution for x from 'filename', reconstructs F1, F2, G,
    evaluates the solution numerically, and plots the curves.
    """

    V1, Km1 = sympy.symbols('V1 Km1', positive=True, real=True)
    V2, Km2 = sympy.symbols('V2 Km2', positive=True, real=True)
    x_min, x_max = sympy.symbols('x_min x_max', real=True, positive=True)
    y_min1, y_max1 = sympy.symbols('y_min1 y_max1', real=True, positive=True)
    y_min2, y_max2 = sympy.symbols('y_min2 y_max2', real=True, positive=True)
    x = sympy.Symbol('x', real=True, positive=True)

    solutions_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # 去除換行符並將字符串解析為字典
            if line.strip():  # 避免空行
                solution = line.strip()  # 安全地解析字符串為Python對象
                solutions_list.append(solution)

    # Reconstruct Sympy expression
    local_dict = {
        "x": x,
        "V1": V1, "Km1": Km1,
        "V2": V2, "Km2": Km2,
        "x_min": x_min, "x_max": x_max,
        "y_min1": y_min1, "y_max1": y_max1,
        "y_min2": y_min2, "y_max2": y_max2
    }
    sol_exprs = [sympy.sympify(sol_str, locals=local_dict) for sol_str in solutions_list]


    # --- 4) Test numeric substitution using provided parameters ---

    param_dict = {
        V1: param_dict["V1"], Km1: param_dict["Km1"],
        V2: param_dict["V2"], Km2: param_dict["Km2"],
        x_min: param_dict["x_min"], x_max: param_dict["x_max"],
        y_min1: param_dict["y_min1"], y_max1: param_dict["y_max1"],
        y_min2: param_dict["y_min2"], y_max2: param_dict["y_max2"]
    }


    x_sol_values = [ sol_expr.subs(param_dict) for sol_expr in sol_exprs]
    # find solution > 0 
    used_sol_i = None
    for idx, x_val in enumerate(x_sol_values):
        # import ipdb; ipdb.set_trace()
        # print(x_val)
        if x_val.evalf() <= param_dict[x_max] and x_val.evalf() >= param_dict[x_min]:
            x_sol_value_f = x_val.evalf()
            if used_sol_i is not None:
                print(f"Warning: multiple solutions in range [{param_dict[x_min]}, {param_dict[x_max]}]. Using the first one found.")
            used_sol_i  = idx
            break
    
    if used_sol_i is None:
        # # Evaluate G at boundaries and mark the smaller one
        # F1: flip x, flip y
        x_norm_1 = (x_max - x)/(x_max - x_min)
        y_norm_1 = (V1 * x_norm_1)/(Km1 + x_norm_1)
        F1_expr = y_min1 + (1 - y_norm_1)*(y_max1 - y_min1)

        # F2: no flip x, flip y
        x_norm_2 = (x - x_min)/(x_max - x_min)
        y_norm_2 = (V2 * x_norm_2)/(Km2 + x_norm_2)
        F2_expr = y_min2 + (1 - y_norm_2)*(y_max2 - y_min2)

        G_expr = F1_expr + F2_expr
        G_min = float(G_expr.subs({**param_dict, **{x: param_dict[x_min]}}))
        G_max = float(G_expr.subs({**param_dict, **{x: param_dict[x_max]}}))
        
        if G_min <= G_max:
            x_sol_value_f = param_dict[x_min]
        else:
            x_sol_value_f = param_dict[x_max]
        used_sol_i = -1

    print(f"[read_and_plot_solution] Loaded solution: {used_sol_i}")
    print("[read_and_plot_solution] Numeric solution => x =", x_sol_value_f)

    return x_sol_value_f




def main(root_dir):
    """
    Main runner that:
      - Iterates over L1, L2, L3, L4
      - Reads coarse_search_results.json AND fine_search_results.json
      - Reads and parses log.txt for targets
      - Filters data
      - Plots data with coarse in blue, fine in red
    """

    for idx, level_name in enumerate(["L1", "L2", "L3", "L4"]):
        if (idx+1) != args.level:
            continue

        level_path = os.path.join(root_dir, level_name)
        if not os.path.isdir(level_path):
            print(f"Skipping {level_path}, not a directory.")
            continue
        
        log_file = os.path.join(level_path, "log.txt")
        if not os.path.exists(log_file):
            print(f"Warning: no log.txt in {level_path}. Skipping.")
            continue
        metrics_log = parse_log_file(log_file)

        # Load coarse & fine
        coarse_file = os.path.join(level_path, "coarse_search_results.json")
        fine_file   = os.path.join(level_path, "fine_search_results.json")

        coarse_data = load_json_results(coarse_file)
        fine_data   = load_json_results(fine_file)

        # # Scale display power by 3x and recompute total power
        # if coarse_data is not None:
        #     coarse_data["display_powers"] = [p * 3.0 for p in coarse_data["display_powers"]]
        #     coarse_data["total_powers"] = [d + r for d, r in zip(coarse_data["display_powers"], coarse_data["render_powers"])]
            
        # if fine_data is not None:
        #     fine_data["display_powers"] = [p * 3.0 for p in fine_data["display_powers"]]
        #     fine_data["total_powers"] = [d + r for d, r in zip(fine_data["display_powers"], fine_data["render_powers"])]

        if coarse_data is None and fine_data is None:
            print(f"No coarse/fine JSON results in {level_path}. Skipping.")
            continue

        if level_name == "L1":
            coarse_data = filter_data_L1(
                coarse_data, metrics_log["target_ssim"], metrics_log["target_psnr"]
            ) if coarse_data else None

            fine_data = filter_data_L1(
                fine_data, metrics_log["target_ssim"], metrics_log["target_psnr"]
            ) if fine_data else None
        else:
            coarse_data = filter_data_Li(
                coarse_data, metrics_log["target_hvs"]
            ) if coarse_data else None

            fine_data = filter_data_Li(
                fine_data, metrics_log["target_hvs"]
            ) if fine_data else None

        # if coarse and fine data are both None
        if coarse_data is None and fine_data is None:
            print(f"No data to plot in {level_path}. Skipping.")
            continue

        # Fit Michaelis-Menten curves to filtered data
        x_data = []
        y_display_data = []
        y_render_data = []

        # Concatenate coarse and fine data
        if coarse_data and len(coarse_data["prune_ratios"]) > 0:
            x_data.extend(coarse_data["prune_ratios"])
            y_display_data.extend(coarse_data["display_powers"])
            y_render_data.extend(coarse_data["render_powers"])
            
        if fine_data and len(fine_data["prune_ratios"]) > 0:
            x_data.extend(fine_data["prune_ratios"])
            y_display_data.extend(fine_data["display_powers"])
            y_render_data.extend(fine_data["render_powers"])


        if len(x_data) > 0:
            # Sort by x values
            indices = np.argsort(x_data)
            x_data = np.array(x_data)[indices]
            y_display_data = np.array(y_display_data)[indices]
            y_render_data = np.array(y_render_data)[indices]
            # compute y_all
            y_all_data = y_display_data + y_render_data
            # huristic best prune ratio is the one with the lowest total power
            huristic_best_prune_ratio = x_data[np.argmin(y_all_data)]
            # Fit display power
            func_display,  param_dict_display = fit_michaelis_menten_with_options(
                x_data,
                y_display_data,
                flip_x=True,
                flip_y=True,
                do_plot=True,
                plot_name=os.path.join(level_path, f"{level_name}_display_mm.png")
            )
            
            # Fit render power
            func_render,  param_dict_render = fit_michaelis_menten_with_options(
                x_data,
                y_render_data,
                flip_x=False,
                flip_y=True,
                do_plot=True,
                plot_name=os.path.join(level_path, f"{level_name}_render_mm.png")
            )

            # Construct parameter dict for find_solution
            param_dict = {
                "V1": param_dict_display["V_fit"],
                "Km1": param_dict_display["Km_fit"],
                "V2": param_dict_render["V_fit"], 
                "Km2": param_dict_render["Km_fit"],
                "x_min": param_dict_display["xmin"],
                "x_max": param_dict_display["xmax"],
                "y_min1": param_dict_display["ymin"],
                "y_max1": param_dict_display["ymax"],
                "y_min2": param_dict_render["ymin"],
                "y_max2": param_dict_render["ymax"]
            }

            # Find optimal solution
            optimal_ratio = find_solution("/workspace/PowerGS/FR_Ours/my_symbolic_solution.txt", param_dict)
            print(f"[{level_name}] Optimal pruning ratio: {optimal_ratio}")


            # Plot total power curve and optimal point
            xx = np.linspace(param_dict["x_min"], param_dict["x_max"], 200)
            yy_display = func_display(xx)
            yy_render = func_render(xx)
            yy_total = yy_display + yy_render

            # save func display and render
            import pickle
            # eval rho = 0-1 and save display and render power pair to a file called display_render_pairs.pkl
            display_render_pairs = []
            for rho in np.linspace(0, 1, 200):
                display_render_pairs.append((func_display(rho), func_render(rho)))
            with open(os.path.join(level_path, "display_render_pairs.pkl"), "wb") as f:
                pickle.dump(display_render_pairs, f)
            
            plt.figure(figsize=(8,6))
            plt.plot(xx, yy_total, 'g-', label="Total Power (fitted)")
            
            # Plot data points using scatter
            if coarse_data and len(coarse_data["prune_ratios"]) > 0:
                plt.scatter(coarse_data["prune_ratios"], coarse_data["total_powers"], c='blue', marker='o', s=50, label="Coarse")
            if fine_data and len(fine_data["prune_ratios"]) > 0:
                plt.scatter(fine_data["prune_ratios"], fine_data["total_powers"], c='red', marker='o', s=50, label="Fine")
            
            # Mark optimal point
            optimal_display = func_display(optimal_ratio)
            optimal_render = func_render(optimal_ratio)
            optimal_total = optimal_display + optimal_render
            plt.plot(optimal_ratio, optimal_total, 'r*', ms=12, label=f"Optimal x={float(optimal_ratio):.3f}")
            
            plt.xlabel("Pruning Ratio")
            plt.ylabel("Total Power")
            plt.title(f"{level_name} - total_powers")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(level_path, f"{level_name}_total_powers.png"))
            plt.close()

            # Save data for future processing
            # cocat fine and coarse 
            sample_x = coarse_data["prune_ratios"] + fine_data["prune_ratios"]
            sample_y = coarse_data["total_powers"] + fine_data["total_powers"]
            data_to_save = {'optimal_ratio': optimal_ratio, 'optimal_total': optimal_total, 'xx': xx, 'yy': yy_total, 'sample_x': sample_x, 'sample_y': sample_y}
            pickle_file = os.path.join(level_path, f"{level_name}_total_powers.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(data_to_save, f)

            # save it to a file under the level path if it is with 20 percent of the huristic best prune ratio
            if abs(optimal_ratio - huristic_best_prune_ratio) < 0.2:
                with open(os.path.join(level_path, "optimal_prune_ratio.txt"), "w") as f:
                    f.write(str(optimal_ratio))
            else: # warning
                print(f"[{level_name}] Warning: Optimal pruning ratio is not close to the huristic best prune ratio.")
            


        print(f"Done processing {level_name}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze and plot PowerGS training results.')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing L1, L2, L3, L4 folders (e.g., /workspace/PowerGS/models/FR_Ours_0.99/m360/bicycle/levels)')
    # add a integer argument for the level
    parser.add_argument('--level', type=int, required=False, default=0,
                        help='The level to analyze (0: L1, 1: L2, 2: L3, 3: L4)')
    
    args = parser.parse_args()
    main(args.root_dir)
