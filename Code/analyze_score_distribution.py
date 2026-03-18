from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Chart" / "score_distribution"
DEFAULT_FILE_PATTERN = "Full_Evaluation*.csv"

TARGET_COLUMNS = [
    "Score_Instruction",
    "Score_Entity",
    "Score_AntiForensics",
]

ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "gbk",
    "latin1",
]


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read {csv_path} with supported encodings") from last_error


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = read_csv_with_fallback(csv_path)

    missing = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    cleaned = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned["Score_Sum"] = cleaned.sum(axis=1)
    cleaned = cleaned.dropna(subset=["Score_Sum"]).copy()
    cleaned["Score_Sum"] = cleaned["Score_Sum"].astype(int)
    cleaned["Source_File"] = csv_path.name
    return cleaned


def build_distribution_table(prepared_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(prepared_frames, ignore_index=True)

    distribution = (
        merged.groupby(["Source_File", "Score_Sum"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Source_File", "Score_Sum"])
    )

    total_per_file = distribution.groupby("Source_File")["Count"].transform("sum")
    distribution["Percentage"] = (distribution["Count"] / total_per_file * 100).round(2)
    return distribution


def build_summary_table(prepared_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(prepared_frames, ignore_index=True)
    summary = (
        merged.groupby("Source_File")["Score_Sum"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    summary.columns = [
        "Source_File",
        "Row_Count",
        "Mean_Score_Sum",
        "Median_Score_Sum",
        "Min_Score_Sum",
        "Max_Score_Sum",
        "Std_Score_Sum",
    ]
    return summary.round(2)


def save_tables(distribution: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution.to_csv(output_dir / "score_sum_distribution_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "score_sum_summary.csv", index=False, encoding="utf-8-sig")

    pivot_count = distribution.pivot(
        index="Score_Sum",
        columns="Source_File",
        values="Count",
    ).fillna(0).astype(int)
    pivot_pct = distribution.pivot(
        index="Score_Sum",
        columns="Source_File",
        values="Percentage",
    ).fillna(0)

    pivot_count.to_csv(output_dir / "score_sum_distribution_count_table.csv", encoding="utf-8-sig")
    pivot_pct.round(2).to_csv(output_dir / "score_sum_distribution_percentage_table.csv", encoding="utf-8-sig")


def save_plot(distribution: pd.DataFrame, output_dir: Path) -> Path:
    pivot_count = distribution.pivot(
        index="Score_Sum",
        columns="Source_File",
        values="Count",
    ).fillna(0).sort_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "P", "X"]

    for index, column in enumerate(pivot_count.columns):
        ax.plot(
            pivot_count.index,
            pivot_count[column],
            marker=markers[index % len(markers)],
            linewidth=2,
            linestyle=line_styles[index % len(line_styles)],
            label=column,
        )

    ax.set_title("Distribution of Score Sum")
    ax.set_xlabel("Score Sum")
    ax.set_ylabel("Count")
    ax.set_xticks(pivot_count.index.tolist())
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path = output_dir / "score_sum_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of Score_Instruction + Score_Entity + Score_AntiForensics"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing CSV files. Default: project root.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_FILE_PATTERN,
        help="Glob pattern used to select CSV files. Default: Full_Evaluation*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write tables and plots. Default: Chart/score_distribution under project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    csv_files = sorted(input_dir.glob(args.pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched '{args.pattern}' under {input_dir}")

    print(f"Found {len(csv_files)} CSV files under {input_dir} matching '{args.pattern}'.")

    prepared_frames = [load_and_prepare(csv_path) for csv_path in csv_files]
    distribution = build_distribution_table(prepared_frames)
    summary = build_summary_table(prepared_frames)

    save_tables(distribution, summary, output_dir)
    plot_path = save_plot(distribution, output_dir)

    print("Analysis complete.")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"  - {(output_dir / 'score_sum_distribution_long.csv')}")
    print(f"  - {(output_dir / 'score_sum_distribution_count_table.csv')}")
    print(f"  - {(output_dir / 'score_sum_distribution_percentage_table.csv')}")
    print(f"  - {(output_dir / 'score_sum_summary.csv')}")
    print(f"  - {plot_path.resolve()}")


if __name__ == "__main__":
    main()
