import torch
import matplotlib.pyplot as plt
from rsl_rl.utils.barrier import relaxed_log_barrier_one_sided, relaxed_barrier_for_interval

# ==== ここに前回作成した barrier 関数群をコピペしてください ====
# relaxed_log_barrier_one_sided_torch
# relaxed_barrier_for_interval_torch
# ===============================================================

def test_plot_one_sided(delta=0.05, k=2):
    # z を -0.2 ～ 0.2 までプロット
    z = torch.linspace(-0.2, 0.2, steps=400)
    B = relaxed_log_barrier_one_sided(z, delta=delta, k=k)

    plt.figure(figsize=(6,4))
    plt.plot(z.numpy(), B.detach().numpy(), label=f"one-sided (δ={delta}, k={k})")
    plt.axvline(x=delta, color='red', linestyle='--', label='delta')
    plt.axhline(y=0, color='gray', linestyle=':')
    plt.title("Relaxed Log Barrier (One-Sided)")
    plt.xlabel("z (margin)")
    plt.ylabel("Barrier value")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_plot_interval(lower=0.0, upper=1.0, delta_frac=0.1, k=2):
    # x を -1.5 ～ 1.5 までプロット
    x = torch.linspace(-1.5, 1.5, steps=400)
    B = relaxed_barrier_for_interval(x, lower=lower, upper=upper, 
                                           delta_frac=delta_frac, k=k)

    plt.figure(figsize=(6,4))
    plt.plot(x.numpy(), B.detach().numpy(), label=f"interval [{lower},{upper}] δ_frac={delta_frac}, k={k}")
    plt.axvline(x=lower, color='red', linestyle='--', label='lower')
    plt.axvline(x=upper, color='blue', linestyle='--', label='upper')
    plt.axhline(y=0, color='gray', linestyle=':')
    plt.title("Relaxed Log Barrier (Interval)")
    plt.xlabel("x")
    plt.ylabel("Barrier value")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 1. 片側バリアをテスト
    test_plot_one_sided(delta=0.05, k=2)

    # 2. 区間制約バリアをテスト
    test_plot_interval(lower=-1.0, upper=1.0, delta_frac=0.1, k=2)
