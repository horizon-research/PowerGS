import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pickle
def generate_test_data(
    n_points=50,
    V_true=1.2,   # 真實 V
    Km_true=0.3,  # 真實 Km
    x_min=0.0,
    x_max=2.0,
    flip_x=False,  # 是否翻轉 X
    flip_y=True,   # 是否翻轉 Y (題目說 y 軸翻轉要加負號，這裡做示範)
    shift_x=0.5,   # 產生資料時，想故意加的 X 偏移
    shift_y=-1.0,  # 故意加的 Y 偏移
    scale_x=2.0,   # 故意把 X 再放大
    scale_y=0.8,   # 故意把 Y 再縮小
    noise_level=0.02  # 雜訊程度
):
    """
    產生測試用的 (x, y) 資料:
      1) 用標準的 Michaelis–Menten 生成理想 y
      2) optional: 加入翻轉 / shift / scale
      3) 加些微雜訊
    回傳: (x_data, y_data) 各為 np.array
    """
    # 1) 先在原始區間 [x_min, x_max] 均勻取點
    x_ideal = np.linspace(x_min, x_max, n_points)

    # 2) Michaelis–Menten (不加任何 offset), 得到 y_ideal
    y_ideal = (V_true * x_ideal) / (Km_true + x_ideal)

    # 3) 加翻轉 / shift / scale
    #    flip_x: 如果要翻轉，就令 x' = (x_max - x_ideal)，否則 x' = x_ideal
    if flip_x:
        x_mod = (x_max - x_ideal)
    else:
        x_mod = x_ideal

    #    flip_y: 如果要翻轉，就令 y' = - y_ideal (題目說 y 要翻轉 = 加負號)
    if flip_y:
        y_mod = - y_ideal
    else:
        y_mod = y_ideal

    #    加 shift & scale
    x_mod = (x_mod + shift_x) * scale_x
    y_mod = (y_mod + shift_y) * scale_y

    # 4) 加入雜訊
    y_noisy = y_mod + noise_level * np.random.randn(n_points)

    return x_mod, y_noisy


def minmax_scale_with_flip(x, flip=False):
    """
    輸入: 一維 array x
    輸出: ( x_norm, x_min, x_max )
      x_norm 落在 [0, 1] ( 若 flip=False ), 或 [1, 0] ( 若 flip=True )
    """
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax == xmin:
        # 避免除以 0
        return np.zeros_like(x), xmin, xmax

    x_norm = (x - xmin) / (xmax - xmin)
    if flip:
        x_norm = 1.0 - x_norm

    return x_norm, xmin, xmax




def mm_func(x, V, Km):
    return (V * x) / (Km + x)


def fit_michaelis_menten(x_norm, y_norm):
    """
    Fit the Michaelis-Menten equation y = (V * x) / (Km + x)
    on the normalized data [0, 1], ensuring V_fit > 0 and Km_fit > 0.
    Returns the optimal parameters popt = [V_fit, Km_fit] and pcov (covariance).
    """
    # Initial guess for [V, Km]
    initial_guess = [1.0, 0.5]
    
    # Set bounds: V > 0, Km > 0
    bounds = ([0, 0], [float('inf'), float('inf')])
    
    # Perform curve fitting with constraints
    popt, pcov = curve_fit(mm_func, x_norm, y_norm, p0=initial_guess, bounds=bounds)
    return popt, pcov


def forward_x_to_norm(x, xmin, xmax, flip_x):
    """
    輸入原始 x，輸出跟擬合時一致的 x_norm
    """
    if xmax == xmin:
        return np.zeros_like(x)
    x_norm = (x - xmin) / (xmax - xmin)
    if flip_x:
        x_norm = 1.0 - x_norm
    return x_norm


def invert_y_from_norm(y_norm, ymin, ymax, flip_y):
    """
    輸入擬合後的 y_norm，轉回原始 y
    """
    if ymax == ymin:
        return np.zeros_like(y_norm)

    if not flip_y:
        # y = ymin + y_norm * (ymax - ymin)
        return ymin + y_norm * (ymax - ymin)
    else:
        # y = ymin + (1 - y_norm) * (ymax - ymin)
        return ymin + (1.0 - y_norm) * (ymax - ymin)


def create_michaelis_menten_function(
    V_fit, Km_fit,
    xmin, xmax, flip_x,
    ymin, ymax, flip_y
):
    """
    回傳一個 callable: F(x) => 預測的 y (在原始空間)
    並提供一個函式把最後的解析式印出來
    """

    def F_of_x(x):
        # 1) 先把 x -> x_norm
        x_n = forward_x_to_norm(x, xmin, xmax, flip_x)
        # 2) y_norm = (V_fit * x_n) / (Km_fit + x_n)
        y_n = mm_func(x_n, V_fit, Km_fit)
        # 3) 最後 invert 回原始空間
        return invert_y_from_norm(y_n, ymin, ymax, flip_y)

    def print_equation():
        """
        輸出最終 F(x) 的完整解析式 (字串)
        """
        # 先寫 x_norm(x) 的字串:
        if not flip_x:
            # x_norm = (x - xmin)/(xmax - xmin)
            xnorm_str = f"((x - {xmin}) / ({xmax} - {xmin}))"
        else:
            # x_norm = 1 - (x - xmin)/(xmax - xmin)
            xnorm_str = f"(1 - (x - {xmin}) / ({xmax} - {xmin}))"

        # 寫 y_norm(x_norm):
        ynorm_str = f"({V_fit} * {xnorm_str}) / ({Km_fit} + {xnorm_str})"

        # 最後組合 y(x)
        if not flip_y:
            # y = ymin + y_norm*(ymax - ymin)
            final_str = (
                f"y = {ymin} + {ynorm_str} * ({ymax} - {ymin})"
            )
        else:
            # y = ymin + (1 - y_norm)*(ymax - ymin)
            #    = ymin + (ymax - ymin) - y_norm*(ymax - ymin)
            final_str = (
                f"y = {ymin} + (1 - {ynorm_str}) * ({ymax} - {ymin})"
            )

        print("Fitted Michaelis–Menten function in original x-space:")
        print(final_str)

    return F_of_x, print_equation



def fit_michaelis_menten_with_options(x, y, flip_x=False, flip_y=True, do_plot=True, plot_name="fit_mm_with_options.png"):
    """
    綜合:
      1) 對 (x, y) 做 min–max scale + 翻轉
      2) 在正規化空間 fit Michaelis–Menten
      3) 把結果還原到原始空間，回傳 F(x)，並印出最後解析式
      4) 可選擇性做繪圖
    """
    # 1) min-max scale + flip
    x_norm, xmin, xmax = minmax_scale_with_flip(x, flip_x)
    y_norm, ymin, ymax = minmax_scale_with_flip(y, flip_y)

    # 2) fit in normalized space
    popt, pcov = fit_michaelis_menten(x_norm, y_norm)
    V_fit, Km_fit = popt
    print("[Fit result] V_fit =", V_fit, ", Km_fit =", Km_fit)

    # 3) 創造一個可直接輸入原始 x，就得到 y 的 function
    F_of_x, print_eq = create_michaelis_menten_function(
        V_fit, Km_fit,
        xmin, xmax, flip_x,
        ymin, ymax, flip_y
    )

    # 4) 印出最終解析式
    # print_eq()

    # 5) (option) 驗證函式 + 繪圖
    if do_plot:
        # 畫原始資料
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, label="Data", color='blue')

        # 畫擬合曲線(在 x 的範圍內均勻取一些點)
        xx = np.linspace(min(x), max(x), 200)
        yy = F_of_x(xx)
        plt.plot(xx, yy, 'r-', label="Fitted MM")

        plt.xlabel("x (original)")
        plt.ylabel("y (original)")
        plt.title("Michaelis–Menten fit (with flip/shift/scale)")
        plt.legend()
        plt.tight_layout()
        save_path = plot_name
        plt.savefig(save_path)
        # Save data for future processing
        data_to_save = {'x': x, 'y': y, 'xx': xx, 'yy': yy, 'save_path': save_path}
        pickle_file = save_path + ".pkl"

        with open(pickle_file, 'wb') as f:
            pickle.dump(data_to_save, f)

    # # Print min/max values and fitting parameters
    # print(f"X range: [{xmin:.3f}, {xmax:.3f}]")
    # print(f"Y range: [{ymin:.3f}, {ymax:.3f}]")
    # print(f"Fitting parameters: V={V_fit:.3f}, Km={Km_fit:.3f}")

    param_dict = {
        "V_fit": V_fit,
        "Km_fit": Km_fit,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax
    }
    return F_of_x,  param_dict
if __name__ == "__main__":
    # ====== 生成測試資料 (可以自行調整參數做實驗) ======
    x_data, y_data = generate_test_data(
        n_points=50,
        V_true=1.2,
        Km_true=0.3,
        x_min=0.0,
        x_max=2.0,
        flip_x=True,   # 在生成測試時，就已經翻轉 x
        flip_y=True,   # 在生成測試時，就已經翻轉 y
        shift_x=0.5,
        shift_y=-0.8,
        scale_x=2.0,
        scale_y=0.8,
        noise_level=0.03
    )

    # ====== 做擬合 ======
    # 注意: 我們要"猜"這筆資料是否有翻轉 x, y，否則擬合會失準
    # 這裡示範: 假設我們"知道" x,y 都有翻轉 => flip_x=True, flip_y=True
    F = fit_michaelis_menten_with_options(
        x_data, 
        y_data, 
        flip_x=True,   # 跟上面生成資料的條件一致
        flip_y=True,
        do_plot=True,
        plot_name="fit_mm_with_options.png"
    )

    # 之後就可以拿 F(x) 來預測任何 x 下的 y
    test_x = 1.0
    print("F(1.0) =", F(test_x))
