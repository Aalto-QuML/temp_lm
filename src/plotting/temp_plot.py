import argparse
import csv
import matplotlib.pyplot as plt
import re
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Ratio vs NLL from metrics CSV.")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/m/home/home0/00/scheufh1/data/Documents/temperature_diffusion/results/metrics_openwebtext_train_slice1000000-1000900_L64_bs4.csv",
        help="Path to the metrics CSV file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/m/home/home0/00/scheufh1/data/Documents/temperature_diffusion/ratio_vs_nll.png",
        help="Path to save the output plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File not found at {args.csv_file}")
        return

    data = []
    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    groups = {}

    for row in data:
        model_id = row["model_id"]

        try:
            # Use ratio_scale_mean as per CSV header
            ratio = float(row["ratio_scale_mean"])
            nll = float(row["mean_nll"])
        except ValueError:
            continue

        group_name = None
        temperature = None

        if model_id == "kuleshov-group/bd3lm-owt-block_size4":
            group_name = "Base Model"
            try:
                temperature = float(row["temperature"])
            except ValueError:
                continue
        else:
            # Clean up path prefix for cleaner legend
            short_name = model_id.replace("/m/cs/work/scheufh1/models/", "")

            # Pattern 1: openwebtext-blocksize4_emp<TEMP>
            match_emp = re.search(r"openwebtext-blocksize4_emp([0-9\.]+)", short_name)

            # Pattern 2: openwebtext_<TEMP>
            match_owt = re.search(r"openwebtext_([0-9\.]+)", short_name)

            if match_emp:
                temp_str = match_emp.group(1)
                try:
                    temperature = float(temp_str)
                    # Replace temp with placeholder for grouping
                    group_name = short_name.replace(f"emp{temp_str}", "emp{T}")[32:80]
                except ValueError:
                    continue
            elif match_owt:
                temp_str = match_owt.group(1)
                try:
                    temperature = float(temp_str)
                    # Replace temp with placeholder for grouping
                    group_name = short_name.replace(
                        f"openwebtext_{temp_str}", "openwebtext_{T}"
                    )[32:80]
                except ValueError:
                    continue
            else:
                continue

        if group_name is not None and temperature is not None:
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append({"temp": temperature, "ratio": ratio, "nll": nll})

    # Plotting
    plt.figure(figsize=(12, 8))

    # Sort groups by name
    sorted_group_names = sorted(groups.keys())

    # Markers
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "x", "+"]

    for i, name in enumerate(sorted_group_names):
        points = groups[name]
        # Sort points by temperature to draw the line in order
        points.sort(key=lambda x: x["temp"])

        ratios = [p["ratio"] for p in points]
        nlls = [p["nll"] for p in points]

        marker = markers[i % len(markers)]

        plt.plot(ratios, nlls, marker=marker, linestyle="-", label=name, markersize=6)

    plt.xlabel("Ratio Scale Mean")
    plt.ylabel("Mean NLL")
    plt.ylim(top=5.0)
    plt.title("Mean NLL vs Ratio Scale Mean")

    # Adjust legend
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", borderaxespad=0.0
    )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(args.output_file, dpi=300)
    print(f"Plot saved to {args.output_file}")


if __name__ == "__main__":
    main()
