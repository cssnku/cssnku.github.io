import re
import json
import random
from tqdm import tqdm
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# =========================
# Similarity functions
# =========================

def longest_common_substring(pwA, pwB):
    lenA, lenB = len(pwA), len(pwB)
    dp = [[0] * (lenB + 1) for _ in range(lenA + 1)]
    max_len = 0

    for i in range(1, lenA + 1):
        for j in range(1, lenB + 1):
            if pwA[i - 1] == pwB[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])

    return max_len


def aligned_target_positions(source, target, min_run_length=2):
    """Return target positions covered by shared contiguous substrings."""
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    positions = set()
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                length = dp[i][j]
                if length >= min_run_length:
                    positions.update(range(j - length, j))
    return positions


def lcs_similarity(pwA, pwB):
    return longest_common_substring(pwA, pwB) / max(len(pwA), len(pwB))


CHAR_BEGIN = '\x01'
CHAR_END = '\x02'
ed_a = [0 for _ in range(65555)]
ed_b = [0 for _ in range(65555)]


def cosine_similarity(a, b):
    a = CHAR_BEGIN + a + CHAR_END
    b = CHAR_BEGIN + b + CHAR_END

    def gram(s, i):
        return ord(s[i]) * 256 + ord(s[i + 1])

    for i in range(len(a) - 1):
        ed_a[gram(a, i)] += 1
    for i in range(len(b) - 1):
        ed_b[gram(b, i)] += 1

    x = y = z = 0
    for i in range(len(a) - 1):
        h1 = gram(a, i)
        if ed_a[h1]:
            y += ed_a[h1] ** 2
            if ed_b[h1]:
                x += ed_a[h1] * ed_b[h1]
                z += ed_b[h1] ** 2
                ed_b[h1] = 0
            ed_a[h1] = 0

    for i in range(len(b) - 1):
        h1 = gram(b, i)
        if ed_b[h1]:
            z += ed_b[h1] ** 2
            ed_b[h1] = 0

    if y == 0 or z == 0:
        return 0

    return math.sqrt(x ** 2 / y / z)


def edit_distance(pwA, pwB):
    lenA, lenB = len(pwA), len(pwB)
    dp = [[0] * (lenB + 1) for _ in range(lenA + 1)]

    for i in range(lenA + 1):
        for j in range(lenB + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif pwA[i - 1] == pwB[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1]
                ) + 1

    return dp[lenA][lenB]


def similarity(pwA, pwB, method="cosine"):
    if method == "edit_distance":
        return 1 - edit_distance(pwA, pwB) / max(len(pwA), len(pwB))
    elif method == "cosine":
        return cosine_similarity(pwA, pwB)
    elif method == "lcs":
        return lcs_similarity(pwA, pwB)
    else:
        raise ValueError("Unsupported method")


# =========================
# Helper functions
# =========================

# def extract_template(pw):
#     res = []
#     for c in pw:
#         if c.isalpha():
#             res.append('L')
#         elif c.isdigit():
#             res.append('D')
#         else:
#             res.append('S')

#     template = []
#     count = 1
#     for i in range(1, len(res) + 1):
#         if i < len(res) and res[i] == res[i - 1]:
#             count += 1
#         else:
#             template.append(f"{res[i-1]}{count}")
#             count = 1

#     return ''.join(template)

def extract_template(pw):
    res = []
    for c in pw:
        if c.isalpha():
            res.append('L')
        elif c.isdigit():
            res.append('D')
        else:
            res.append('S')
    return ''.join(res)  # do not collapse runs


def extract_bigrams(pw):
    if len(pw) < 2:
        return {pw}
    return {pw[i:i+2] for i in range(len(pw) - 1)}


# =========================
# User-level metrics (CharJaccard / MinEditDist removed)
# =========================

def compute_user_metrics_v2(pwds, max_pw_per_user=20):

    pwds = list(set(pwds))

    if len(pwds) > max_pw_per_user:
        pwds = random.sample(pwds, max_pw_per_user)

    k = len(pwds)
    if k < 2:
        return None

    # =========================
    # Similarity matrix
    # =========================
    sim_matrix = [[0]*k for _ in range(k)]
    sims = []

    for i in range(k):
        for j in range(i+1, k):
            s = cosine_similarity(pwds[i], pwds[j])
            sim_matrix[i][j] = s
            sim_matrix[j][i] = s
            sims.append(s)

    mean_sim = sum(sims) / len(sims)
    max_sim = max(sims)
    var_sim = np.var(sims) if k >= 3 else None

    pivot_sim = None
    if k >= 3:
        centrality = [
            sum(sim_matrix[i][j] for j in range(k) if j != i) / (k - 1)
            for i in range(k)
        ]
        pivot_sim = max(centrality)

    # # =========================
    # # TemplateCons
    # # =========================
    # templates = [extract_template(pw) for pw in pwds]
    # cnt = Counter(templates)
    # probs = np.array(list(cnt.values())) / k
    # entropy = -np.sum(probs * np.log(probs + 1e-12))
    # template_cons = 1 - entropy / np.log(k)

    # # =========================
    # # LenCV
    # # =========================
    # lengths = [len(pw) for pw in pwds]
    # len_cv = np.std(lengths) / (np.mean(lengths) + 1e-12)

    # =========================
    # TemplateCons (strict paper definition)
    # =========================
    templates = [extract_template(pw) for pw in pwds]
    cnt = Counter(templates)
    probs = np.array(list(cnt.values())) / k
    entropy = -np.sum(probs * np.log2(probs + 1e-12))  # log2！
    template_cons = 1 - entropy / np.log2(k)            # log2！

    # =========================
    # LenCV
    # =========================
    lengths = [len(pw) for pw in pwds]
    mu = np.mean(lengths)
    if mu < 1e-12:
        len_cv = 0.0
    else:
        sigma = np.std(lengths, ddof=0)  # population std
        len_cv = sigma / mu

    # =========================
    # MaxLCSCov
    # =========================
    max_lcs_cov = 0
    for i in range(k):
        for j in range(i+1, k):
            lcs = longest_common_substring(pwds[i], pwds[j])
            cov = lcs / min(len(pwds[i]), len(pwds[j]))
            max_lcs_cov = max(max_lcs_cov, cov)

    # =========================
    # MS-PosCopyRate (multi-source positional copy rate)
    # =========================
    # Randomly select one target password
    target = random.choice(pwds)
    others = [pw for pw in pwds if pw != target]

    covered_positions = set()
    for src_pw in others:
        covered_positions.update(aligned_target_positions(src_pw, target))
    ms_pos_copy_rate = len(covered_positions) / len(target) if target else 0.0

    # =========================
    # MutRatio
    # =========================
    idx = pwds.index(target)
    others_mut = pwds[:idx] + pwds[idx+1:]

    best_pwd = None
    best_sim = -1
    for pw in others_mut:
        sim = cosine_similarity(pw, target)
        if sim > best_sim:
            best_sim = sim
            best_pwd = pw

    lev = edit_distance(best_pwd, target)
    mut_ratio = lev / len(target)

    return {
        "k": k,
        "MeanSim": mean_sim,
        "MaxSim": max_sim,
        "VarSim": var_sim,
        "PivotSim": pivot_sim,
        "TemplateCons": template_cons,
        "LenCV": len_cv,
        "MaxLCSCov": max_lcs_cov,
        "MS_PosCopyRate": ms_pos_copy_rate,
        "MutRatio": mut_ratio,
        # CharJaccard and MinEditDist removed
    }

# =========================
# Data parsing
# =========================

def parse_files_grouped(file_paths,
                        total_lines_list,
                        final_sample_size=50000):

    # =========================
    # Reservoir pools
    # =========================
    reservoirs = {
        "k=2": [],
        "k=3": [],
        "k=4": [],
        "k=5": [],
        "k>=6": []
    }

    seen_counts = {k: 0 for k in reservoirs}

    # =========================
    # Multi-file mixed reading
    # =========================
    for file_path, total_lines in zip(file_paths, total_lines_list):

        print(f"\n[Reading] {file_path}")

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, total=total_lines):

                parts = re.split(r'\s*\t\s*', line.strip())
                if len(parts) != 2:
                    continue

                try:
                    pwds = json.loads(parts[1])
                except:
                    continue

                pwds = list(set(pwds))  # deduplicate
                k = len(pwds)

                # =========================
                # Grouping
                # =========================
                if k == 2:
                    group = "k=2"
                elif k == 3:
                    group = "k=3"
                elif k == 4:
                    group = "k=4"
                elif k == 5:
                    group = "k=5"
                elif k >= 6:
                    group = "k>=6"
                else:
                    continue

                seen_counts[group] += 1
                n = seen_counts[group]

                # =========================
                # Reservoir sampling (shared across files)
                # =========================
                if len(reservoirs[group]) < final_sample_size:
                    reservoirs[group].append(pwds)
                else:
                    r = random.randint(0, n - 1)
                    if r < final_sample_size:
                        reservoirs[group][r] = pwds

    # =========================
    # Output sampling results
    # =========================
    print("\n[Reservoir Done - Mixed]")
    for g in reservoirs:
        print(g, len(reservoirs[g]))

    # =========================
    # Post-processing
    # =========================
    groups = {k: [] for k in reservoirs}

    for g in reservoirs:
        print(f"Processing {g} ...")

        for pwds in tqdm(reservoirs[g], desc=f"{g}"):
            metrics = compute_user_metrics_v2(pwds)
            if metrics:
                groups[g].append(metrics)

    return groups


def plot_by_k(groups, prefix="./pic_k_final/Similarity"):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    from matplotlib.pyplot import MultipleLocator

    COLOR1 = '#9BEB9B'
    COLOR2 = '#feccca'
    COLOR3 = '#909ae2'
    COLOR4 = '#ffebea'
    COLOR5 = '#F25900'

    Pattern1 = 'x'
    Pattern2 = '.'
    Pattern3 = '\\\\'
    Pattern4 = '/'
    Pattern5 = '*'

    Edge_color = 'black'
    FONT_size = 20

    labels = ["k=2", "k=3", "k=4", "k=5", "k>=6"]
    colors = [COLOR1, COLOR2, COLOR3, COLOR4, COLOR5]
    patterns = [Pattern1, Pattern2, Pattern3, Pattern4, Pattern5]

    # =========================
    # Final metric list: CharJaccard / MinEditDist removed, MS_PosCopyRate added
    # =========================
    keys = [
        "MeanSim", "MaxSim", "VarSim", "PivotSim",
        "TemplateCons", "LenCV", "MaxLCSCov",
        "MS_PosCopyRate",
        "MutRatio",
    ]

    for key in keys:
        print(f"\n{'='*80}")
        print(f"Metric: {key}")
        print(f"{'='*80}")
        
        plt.figure(figsize=(10, 5))

        if key == "VarSim":
            bins = np.linspace(0, 0.2, 6)
        else:
            # bins = np.linspace(0, 1, 6)
            # 0..1 range with a tiny overshoot so 1.0 is included
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]

        if key == "VarSim":
            x_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        else:
            x_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        x = np.arange(len(x_labels))

        total_width, n = 0.8, len(labels)
        width = total_width / n
        x = x - (total_width - width) / 2

        all_percentages = []
        for label in labels:
            data = [r[key] for r in groups[label] if r[key] is not None]
            if len(data) == 0:
                all_percentages.append([0]*(len(bins)-1))
                continue
            counts, _ = np.histogram(data, bins=bins)
            percentages = counts / len(data)
            all_percentages.append(percentages)

        # =========================
        # Print the exact percentage of each bar
        # =========================
        for idx_k, group_k in enumerate(labels):
            print(f"\nGroup {group_k}:")
            perc = all_percentages[idx_k]
            for idx_bin, interval in enumerate(x_labels):
                pct = perc[idx_bin] * 100
                print(f"  Bin {interval}: {pct:.2f}%")

        bars = []
        for i, percentages in enumerate(all_percentages):
            p = plt.bar(
                x + (i - (n-1)/2) * width,
                percentages,
                width=width - 0.02,
                label=labels[i],
                color=colors[i],
                hatch=patterns[i],
                edgecolor=Edge_color
            )
            bars.append(p)

        plt.xticks(x, x_labels)
        plt.xlabel(key, fontsize=FONT_size)
        plt.ylabel("Proportion", fontsize=FONT_size)

        plt.xticks(size=FONT_size)
        plt.yticks(size=FONT_size)

        ax = plt.gca()
        y_major_locator = MultipleLocator(0.2)
        ax.yaxis.set_major_locator(y_major_locator)

        def fun(tmp, position):
            return '%.0f' % (100 * tmp) + '%'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(fun))

        plt.legend(
            bars, labels, loc='best', ncol=2,
            fontsize=FONT_size, frameon=False
        )

        plt.tight_layout()
        plt.savefig(f"{prefix}_{key}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.close()


# =========================
# Main
# =========================

if __name__ == "__main__":
    from run_metrics import main
    main()
