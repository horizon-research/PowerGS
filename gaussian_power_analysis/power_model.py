import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------------
# Define constants outside of the function:
# ---------------------------------------------------------------------------------
FLOPs_Proj_Culling = 47     # FLOPs per point in the project + culling phase
FLOPs_3Dto2D = 169           # FLOPs for 3D-to-2D transformation
FLOPs_OBB_Inter_Test = 48     # FLOPs for OBB intersection test
FLOPs_AABB_Inter_Test = 39
FLOPs_prep_OBB = 38
FLOPs_SHs = 133               # FLOPs for shading calculations (example)
FLOPs_CompAlpha = 13         # FLOPs for alpha blending computations + Transmittance
FLOPs_InterColor = 11         # FLOPs for color interpolation computations (share a t)
FLOPs_Final_blend = 6
# ---------------------------------------------------------------------------------
SRAM_per_byte_energy = 0.24    # Energy cost of SRAM access per byte (in pJ)
DRAM_per_byte_energy = 10.88   # Energy cost of DRAM access per byte (in pJ)
FLOPs_energy = 0.53            # Energy cost of a single FLOP (in pJ)
FPS = 60                       # Frames per second


def hardware_power_modeling(
    point_statistics: torch.Tensor,
    pixel_statistics: torch.Tensor
):
    """
    This function calculates the hardware cost model in three stages:
      1) Preprocessing
      2) Sorting
      3) Rendering

    Args:
        point_statistics (torch.Tensor): shape (5, P), containing:
            [ Pcull, N_inter, P_inter, P_used, P_obb_count ] per point.
        pixel_statistics (torch.Tensor): shape (4, H, W), containing:
            [ Tinter, Tload, pixel_bf_term, pixel_used ] per pixel.

    Returns:
        dict: a dictionary containing DRAM, SRAM, and FLOPs usage for each stage and total:
            {
                "preprocessing_dram",
                "preprocessing_sram",
                "preprocessing_flops",
                "sorting_dram",
                "sorting_sram",
                "sorting_flops",
                "rendering_dram",
                "rendering_sram",
                "rendering_flops",
                "total_dram",
                "total_sram",
                "total_flops"
            }
    """

    # -------------------------------------------------------------------------
    # 1. Preprocessing Stage
    # -------------------------------------------------------------------------
    # P = number of input points (Pin)
    P = point_statistics.shape[1]

    # Sum each row across all points:
    # Pcull: how many points were culled
    # N_inter: total #inter
    # P_inter: sum of P_inter
    # P_used: total #used_p
    Pcull   = point_statistics[0].sum().item()
    N_inter = point_statistics[1].sum().item()
    P_inter = point_statistics[2].sum().item()
    P_used  = point_statistics[3].sum().item()
    OBB_used = (point_statistics[4]>1).sum().item()
    OBB_sum = point_statistics[4].sum().item()
    # import ipdb; ipdb.set_trace()

    # DRAM cost in Preprocessing (in bytes)
    #   #DRAM(bytes) = 6 * P + 12 * Pcull + 4 * N_inter + 10 * P_inter
    preprocessing_dram = (
        6 * P
        + 12 * Pcull
        + 4 * N_inter
        + 10 * P_inter
    )

    # SRAM cost in Preprocessing (in bytes)
    #   #SRAM = 0
    preprocessing_sram = 0

    # FLOPs in Preprocessing
    #   FLOPs = P * FLOPs_Proj_Culling
    #           + Pcull * (FLOPs_3Dto2D + FLOPs_OBB_Inter_Test)
    preprocessing_flops = (
        P * FLOPs_Proj_Culling
        + Pcull * (FLOPs_3Dto2D + FLOPs_AABB_Inter_Test)
        + OBB_used * FLOPs_prep_OBB + OBB_sum * FLOPs_OBB_Inter_Test
    )

    # -------------------------------------------------------------------------
    # 2. Sorting Stage
    # -------------------------------------------------------------------------
    # According to the problem statement, we need to:
    #   (1) reflect-pad pixel_statistics to make H and W multiples of 16
    #   (2) reshape into tiles (16x16), compute tile-level averages
    #   (3) for each tile, calculate #DRAM, #SRAM, #FLOPs using the given formulas
    _, H, W = pixel_statistics.shape

    # Calculate required padding for H and W to be multiples of 16
    pad_h = (16 - (H % 16)) if (H % 16) != 0 else 0
    pad_w = (16 - (W % 16)) if (W % 16) != 0 else 0

    # Reflect padding on the right (W dimension) and bottom (H dimension)
    pixel_stats_padded = F.pad(
        pixel_statistics, 
        (0, pad_w, 0, pad_h),  # (pad_left, pad_right, pad_top, pad_bottom)
        mode='reflect'
    )

    # Padded shape
    _, H_pad, W_pad = pixel_stats_padded.shape
    assert H_pad % 16 == 0, "Padded height must be multiple of 16."
    assert W_pad % 16 == 0, "Padded width must be multiple of 16."

    # Reshape to (4, nH, 16, nW, 16) where nH=H_pad//16, nW=W_pad//16
    pixel_stats_tiles = pixel_stats_padded.view(
        4, 
        H_pad // 16, 16,
        W_pad // 16, 16
    )
    # FIRST TRANSPOSE
    pixel_stats_tiles = pixel_stats_tiles.permute(0, 1, 3, 2, 4)

    # merge final two dimensions
    pixel_stats_tiles = pixel_stats_tiles.reshape(4, H_pad // 16, W_pad // 16, 16*16).float()


    # import ipdb; ipdb.set_trace()
    # Compute mean for each 16x16 tile across dimensions 2 and 4
    tile_means = pixel_stats_tiles.mean(dim=-1)

    # Extract the tile-level averages for Tinter and Tload
    # Tinter, Tload, Tbf_term, Tused ] per pixel.
    T_inter = tile_means[0]  # shape (nH, nW)
    T_load  = tile_means[1]  # shape (nH, nW)

    # We need to compute the ceiling of (T_load / 256) in a vectorized manner:
    T_load_div_256_ceiled = torch.ceil(T_load / 256.0)  # shape (nH, nW)


    # According to the problem statement:
    #   #DRAM(bytes)  = 4 * T_inter
    #   #SRAM(bytes)  = 2 * T_inter + 4864 * ceil(T_load/256)
    #   #FLOPs        = 7 * T_inter + 1024 * ceil(T_load/256)
    # Round1 ~ Ceil(log8(T_inter / 256))
    round1 = torch.ceil(torch.log2(T_inter / 256) / 3)
    round1[T_inter < 256] = 0
    # check if round any lower than 1, if so warning


    dram_tiles  = 4 * T_inter
    sram_tiles  = round1 * 8 * T_inter + 1024 * T_load_div_256_ceiled
    flops_tiles = 0

    # Finally, sum over all tiles to get the totals.
    sorting_dram_total  = dram_tiles.sum().item()
    sorting_sram_total  = sram_tiles.sum().item()
    sorting_flops_total = 0

    # -------------------------------------------------------------------------
    # 3. Rendering Stage
    # -------------------------------------------------------------------------
    # We need the sums for Tload, P_used, Tbf_term, Tused:
    # #load    = sum(Tload)
    # #used_p  = sum(P_used)
    # #bf_term = sum(Tbf_term)
    # #used    = sum(Tused)
    used_p = P_used
    bf_term_sum = pixel_statistics[2].sum().item()
    used_sum = pixel_statistics[3].sum().item()

    # #DRAM(bytes) = ceil(#load / 256) * 256 * 19
    #               + #used_p * 103
    #               + (H * W * 3)
    rendering_dram = (
        (T_load * 19).sum().item()
        + used_p * 103
        + (H * W * 3)
    )


    # #SRAM(bytes) = ceil(#load / 256) * 256 * 19 * 2
    rendering_sram = (
        (T_load * 19 * 2).sum().item()
    )

    # #FLOPs = (FLOPs_SHs * #used_p)
    #          + (FLOPs_CompAlpha * #bf_term)
    #          + (FLOPs_InterColor * #used)
    rendering_flops = (
        (FLOPs_SHs * used_p)
        + (FLOPs_CompAlpha * bf_term_sum)
        + (FLOPs_InterColor * used_sum)
        + (FLOPs_Final_blend * H * W)
    )

    # -------------------------------------------------------------------------
    # 4. Sum up total costs
    # -------------------------------------------------------------------------
    total_dram = preprocessing_dram + sorting_dram_total + rendering_dram
    total_sram = preprocessing_sram + sorting_sram_total + rendering_sram
    total_flops = preprocessing_flops + sorting_flops_total + rendering_flops


    # Power
    preprocessing_energy = preprocessing_flops * FLOPs_energy + preprocessing_dram * DRAM_per_byte_energy + preprocessing_sram * SRAM_per_byte_energy
    preprocessing_power = preprocessing_energy * FPS * 1e-12
    sorting_energy = sorting_flops_total * FLOPs_energy + sorting_dram_total * DRAM_per_byte_energy + sorting_sram_total * SRAM_per_byte_energy
    sorting_power = sorting_energy * FPS * 1e-12
    rendering_energy = rendering_flops * FLOPs_energy + rendering_dram * DRAM_per_byte_energy + rendering_sram * SRAM_per_byte_energy
    
    rendering_power = rendering_energy * FPS * 1e-12

    total_dram_power = (preprocessing_dram * DRAM_per_byte_energy + \
                        sorting_dram_total * DRAM_per_byte_energy + rendering_dram * DRAM_per_byte_energy) * FPS * 1e-12
    total_sram_power = (preprocessing_sram * SRAM_per_byte_energy \
        + sorting_sram_total * SRAM_per_byte_energy + rendering_sram * SRAM_per_byte_energy) * FPS * 1e-12
    total_flops_power = (preprocessing_flops * FLOPs_energy \
        + sorting_flops_total * FLOPs_energy + rendering_flops * FLOPs_energy)* FPS * 1e-12
    total_power = preprocessing_power + sorting_power + rendering_power
    
    # import ipdb; ipdb.set_trace()
    # Package results into a dictionary
    results = {
        "preprocessing_dram": preprocessing_dram,
        "preprocessing_sram": preprocessing_sram,
        "preprocessing_flops": preprocessing_flops,

        "sorting_dram": sorting_dram_total,
        "sorting_sram": sorting_sram_total,
        "sorting_flops": sorting_flops_total,

        "rendering_dram": rendering_dram,
        "rendering_sram": rendering_sram,
        "rendering_flops": rendering_flops,

        "total_dram": total_dram,
        "total_sram": total_sram,
        "total_flops": total_flops,

        "preprocessing_power": preprocessing_power,
        "sorting_power": sorting_power,
        "rendering_power": rendering_power,

        "total_power": total_power,
        "total_dram_power": total_dram_power,
        "total_sram_power": total_sram_power,
        "total_flops_power": total_flops_power


    }

    # import ipdb; ipdb.set_trace()
    return results


if __name__ == "__main__":
    # Example usage:

    # Example point_statistics with shape (4, 10):
    point_stats_example = torch.tensor([
        [1,2,1,3,0,0,2,1,1,1],  # Pcull
        [0,1,2,1,1,0,0,2,0,1],  # N_inter
        [3,1,0,0,2,1,0,1,1,1],  # P_inter
        [1,1,1,1,1,1,0,0,0,0],  # P_used
    ], dtype=torch.float)

    # Example pixel_statistics with shape (4, 8, 8), randomly generated
    pixel_stats_example = torch.rand(4, 8, 8) * 5

    # Call the function with our constants
    res = hardware_power_modeling(
        point_stats_example,
        pixel_stats_example
    )

    # Print results
    for key, val in res.items():
        print(f"{key}: {val}")
