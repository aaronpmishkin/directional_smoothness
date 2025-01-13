#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import qr


def f(x, A, b):
    return 0.5 * np.dot(np.dot(x.T, A), x) - np.dot(b.T, x)


def gradient(x, A, b):
    return np.dot(A, x) - b


def dir_smoothness(grad, prev_grad, x, prev_x):
    return np.linalg.norm(grad - prev_grad) / np.linalg.norm(x - prev_x)


def gradient_descent_helper(A, b, num_iterations, x_init, eta_callback):
    d = A.shape[0]
    x = x_init.copy()
    f_values = [f(x, A, b)]
    m_values = [1.0]
    x_prev = None
    grad_prev = None
    step_sizes = []

    for i in range(num_iterations):
        grad = gradient(x, A, b)
        if grad_prev is not None:
            m = dir_smoothness(grad, grad_prev, x, x_prev)
            m_values.append(m)
        grad_prev = grad.copy()
        x_prev = x.copy()

        # Compute learning rate using provided callback function
        eta = eta_callback(grad, A)
        step_sizes.append(eta)
        x = x - eta * grad
        f_values.append(f(x, A, b))

    return x, f_values[:-1], step_sizes, m_values


def gradient_descent(A, b, eta, num_iterations, x_init):
    return gradient_descent_helper(
        A, b, num_iterations, x_init, lambda grad, A: eta
    )


def gradient_descent_new_stepsize(A, b, num_iterations, x_init):
    def new_stepsize_callback(grad, A):
        return 0.5 * np.linalg.norm(grad) / np.linalg.norm(A.dot(grad))

    return gradient_descent_helper(
        A, b, num_iterations, x_init, new_stepsize_callback
    )


def gradient_descent_new_stepsize_second(A, b, num_iterations, x_init):
    def new_stepsize_callback(grad, A):
        return (np.linalg.norm(grad) ** 2) / (np.dot(grad.T, np.dot(A, grad)))

    return gradient_descent_helper(
        A, b, num_iterations, x_init, new_stepsize_callback
    )


def generate_power_law_distribution(alpha, L, size):
    # Generate random numbers from a power law distribution
    power_law_values = np.random.power(alpha, size=size)

    # Scale the values to the desired range [0, L]
    scaled_values = power_law_values * L

    # Sort the values
    sorted_values = np.sort(scaled_values)

    return sorted_values


d = 300  # Problem dimension
num_iterations = 20000
num_random_trials = 20

average_f_values1_gd = np.zeros(num_iterations)
average_f_values2_gd = np.zeros(num_iterations)
average_f_values_new_gd = np.zeros(num_iterations)
average_f_values_new2_gd = np.zeros(num_iterations)

average_m_values1 = np.zeros(num_iterations)
average_m_values2 = np.zeros(num_iterations)
average_m_values_new_stepsize = np.zeros(num_iterations)
average_m_values_new2_stepsize = np.zeros(num_iterations)

m_values1_all = []
m_values2_all = []
m_values_new_stepsize_all = []
m_values_new2_stepsize_all = []

f_values1_gd_all = []
f_values2_gd_all = []
f_values_new_gd_all = []
f_values_new2_gd_all = []

step_size_avg = [0] * num_iterations
step_size_avg2 = [0] * num_iterations
avg_mu = 0.0
avg_L = 0.0


# Parameters for the power law distribution
alpha = 3.0  # The shape parameter (adjust as needed)
L = 1000.0  # The upper bound of the range [0, L]
# Generate the sorted list of numbers following a power law distribution
sorted_power_law_values = L - generate_power_law_distribution(alpha, L, d)

for _ in range(num_random_trials):
    eig = sorted_power_law_values
    # eig = L * np.exp(-np.linspace(0, 100, num=d))
    # eig = np.linspace(L, 0, num=d)
    # eig = np.array([1e-7] * d)
    # eig[0] = L
    eigenvals = np.diag(eig)
    H = np.random.randn(d, d)
    Q, _ = qr(H)
    A = Q.dot(eigenvals.dot(Q.T))
    b = np.random.rand(d)
    eig[-1] = 10e-6
    eig[0] = L
    avg_mu += eig[-1] / num_random_trials
    avg_L += eig[0] / num_random_trials

    x_init = np.random.rand(d) * 10

    minimizer = np.linalg.pinv(A).dot(b)
    min_value = f(minimizer, A, b)

    _, f_values_eta1, _, m_values1 = gradient_descent(
        A, b, 1 / eig[0], num_iterations, x_init
    )
    _, f_values_eta2, _, m_values2 = gradient_descent(
        A, b, 2 / eig[0], num_iterations, x_init
    )
    (
        _,
        f_values_new_stepsize,
        step_sizes,
        m_values_new_stepsize,
    ) = gradient_descent_new_stepsize(A, b, num_iterations, x_init)

    # gradient_descent_new_stepsize_second
    (
        _,
        f_values_new2_stepsize,
        step_sizes2,
        m_values_new2_stepsize,
    ) = gradient_descent_new_stepsize_second(A, b, num_iterations, x_init)
    print(f_values_new_stepsize[0], f_values_new2_stepsize[0])

    step_size_avg += np.array(step_sizes)
    step_size_avg2 += np.array(step_sizes2)

    average_f_values1_gd += np.array(f_values_eta1 - min_value)
    average_f_values2_gd += np.array(f_values_eta2 - min_value)
    average_f_values_new_gd += np.array(f_values_new_stepsize - min_value)
    average_f_values_new2_gd += np.array(f_values_new2_stepsize - min_value)

    average_m_values1 += np.array(m_values1 / eig[0])
    average_m_values2 += np.array(m_values2 / eig[0])
    average_m_values_new_stepsize += np.array(m_values_new_stepsize / eig[0])
    average_m_values_new2_stepsize += np.array(
        f_values_new2_stepsize - min_value
    )

    f_values1_gd_all.append(f_values_eta1)
    f_values2_gd_all.append(f_values_eta2)
    f_values_new_gd_all.append(f_values_new_stepsize)
    f_values_new2_gd_all.append(f_values_new2_stepsize)

    m_values1_all.append(m_values1)
    m_values2_all.append(m_values2)
    m_values_new_stepsize_all.append(m_values_new_stepsize)
    m_values_new2_stepsize_all.append(m_values_new_stepsize)


# plot parameters
marker_styles = ["o", "s", "v", "X", "D", "^", "D", "p", "o", "x", "s"]

line_colors = [
    "#000000",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#8c564b",
    "#17becf",
    "#556B2F",
    "#FFFF00",
    "#191970",
]
markersize = 8


label_fs = 26
title_fs = 30
legend_fs = 26
ticks_fs = 22

line_width = 6

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(19.2, 4.3))

step_size_avg /= num_random_trials
step_size_avg2 /= num_random_trials
ax2.plot(
    range(num_iterations),
    step_size_avg,
    label="$1/D_k$",
    color=line_colors[1],
    # marker=marker_styles[1],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
ax2.plot(
    range(num_iterations),
    step_size_avg2,
    label="$1/A_k$",
    color=line_colors[3],
    # marker=marker_styles[1],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
ax2.plot(
    range(num_iterations),
    [1.0 / avg_L] * num_iterations,
    label="$1/L$",
    color=line_colors[2],
    # marker=marker_styles[2],
    # markevery=0.2,
    linestyle="dashed",
    linewidth=line_width,
)
ax2.plot(
    range(num_iterations),
    [2.0 / avg_L] * num_iterations,
    label="$2/L$",
    color=line_colors[5],
    linestyle="dashed",
    linewidth=line_width,
)
ax2.set_xscale("log")
ax2.set_yscale("log")
# plt.ylabel("Step-sizes", fontsize=label_fs)
ax2.set_xlabel("Iteration", fontsize=label_fs)
ax2.tick_params(labelsize=ticks_fs)
ax2.set_title("Adapted Step-Sizes", fontsize=title_fs)

# plt.show()
average_m_values1 /= num_random_trials
average_m_values2 /= num_random_trials
average_m_values_new_stepsize /= num_random_trials
average_m_values_new2_stepsize /= num_random_trials

ax1.plot(
    range(num_iterations),
    average_m_values1 * avg_L,
    label="$1/L$",
    color=line_colors[2],
    # marker=marker_styles[2],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
# plt.plot(range(num_iterations), average_m_values2, label="Stepsize 2/L")
ax1.plot(
    range(num_iterations),
    average_m_values_new_stepsize * avg_L,
    label="$1/D_k$",
    color=line_colors[1],
    # marker=marker_styles[1],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
ax1.plot(
    range(num_iterations),
    average_m_values_new2_stepsize * avg_L,
    label="$1/A_k$",
    color=line_colors[3],
    # marker=marker_styles[3],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Iteration", fontsize=label_fs)
ax1.set_ylabel("", fontsize=label_fs)
ax1.set_title("Point-Wise Smoothness", fontsize=title_fs)
ax1.tick_params(labelsize=ticks_fs)

average_f_values1_gd /= num_random_trials
average_f_values2_gd /= num_random_trials
average_f_values_new_gd /= num_random_trials
average_f_values_new2_gd /= num_random_trials

# Calculate standard deviation for error bars
std_dev_gd1 = np.std(f_values1_gd_all, axis=0)
std_dev_gd2 = np.std(f_values2_gd_all, axis=0)
std_dev_new_gd = np.std(f_values_new_gd_all, axis=0)
std_dev_new2_gd = np.std(f_values_new_gd_all, axis=0)


ax0.plot(
    range(num_iterations),
    average_f_values1_gd,
    label="$1/L$",
    color=line_colors[2],
    # marker=marker_styles[2],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
# plt.plot(range(num_iterations), average_f_values2_gd, label="GD with stepsize 2/L")
ax0.plot(
    range(num_iterations),
    average_f_values_new_gd,
    label="$1/D_k$",
    color=line_colors[1],
    # marker=marker_styles[1],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)
ax0.plot(
    range(num_iterations),
    average_f_values_new2_gd,
    label="$1/A_k$",
    color=line_colors[3],
    # marker=marker_styles[3],
    # markevery=0.2,
    # markersize=markersize,
    linewidth=line_width,
)

# Plot the average trajectories with error bars

ax0.fill_between(
    range(num_iterations),
    average_f_values1_gd - std_dev_gd1,
    average_f_values1_gd + std_dev_gd1,
    alpha=0.4,
    color=line_colors[4],
)
# plt.fill_between(
#     range(num_iterations),
#     average_f_values2_gd - std_dev_gd2,
#     average_f_values2_gd + std_dev_gd2,
#     alpha=0.7,
# )


ax0.fill_between(
    range(num_iterations),
    average_f_values_new_gd - std_dev_new_gd,
    average_f_values_new_gd + std_dev_new_gd,
    alpha=0.4,
    color=line_colors[6],
)

ax0.fill_between(
    range(num_iterations),
    average_f_values_new2_gd - std_dev_new2_gd,
    average_f_values_new2_gd + std_dev_new2_gd,
    alpha=0.4,
    color=line_colors[6],
)
ax0.set_xscale("log")
ax0.set_xlabel("Iteration", fontsize=label_fs)
ax0.set_ylabel("", fontsize=label_fs)
ax0.set_yscale("log")
ax0.set_title("Optimality Gap", fontsize=title_fs)
ax0.tick_params(labelsize=ticks_fs)

handles0, labels0 = ax0.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles0 + handles2
labels = labels0 + labels2

handles_to_plot, labels_to_plot = [], []
for i, label in enumerate(labels):
    if label not in labels_to_plot:
        handles_to_plot.append(handles[i])
        labels_to_plot.append(label)

legend = fig.legend(
    handles=handles_to_plot,
    labels=labels_to_plot,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=6,
    fontsize=legend_fs,
    frameon=False,
)

plt.tight_layout()
fig.subplots_adjust(
    wspace=0.3,
    hspace=0.15,
    bottom=0.34,
)


plt.savefig("quadratic.pdf", bbox_inches="tight", dpi=200)
