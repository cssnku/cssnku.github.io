#!/usr/bin/env python3
# Run the nine-metric behavior analysis on a small user-level password set file.
# Input format: one line per user: <user_id>\t["pw1","pw2",...]
import argparse, os
import analyze_behavior

def main():
    ap = argparse.ArgumentParser(description="Nine-metric multi-password behavior analysis")
    ap.add_argument("--input", default="data/mini/naz_mini_users.txt")
    ap.add_argument("--output-prefix", default="results/metrics/Similarity")
    ap.add_argument("--sample-size", type=int, default=100)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    nlines = sum(1 for _ in open(args.input, encoding="utf-8"))

    groups = analyze_behavior.parse_files_grouped(
        [args.input], [nlines], final_sample_size=args.sample_size
    )
    for k in groups:
        print(k, len(groups[k]))
    analyze_behavior.plot_by_k(groups, prefix=args.output_prefix)
    print("Plots written to", out_dir + "/Similarity_*.pdf")

if __name__ == "__main__":
    main()
