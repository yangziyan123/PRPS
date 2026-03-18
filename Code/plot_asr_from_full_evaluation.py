from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT_DIR / "Chart"
OUTPUT_PNG = CHART_DIR / "asr_by_model.png"
OUTPUT_CSV = CHART_DIR / "asr_summary.csv"

FILE_PATTERN = "Full_Evaluation*.csv"
ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]


def read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read file: {file_path}") from last_error


def extract_model_name(file_path: Path) -> str:
    stem = file_path.stem
    match = re.search(r"Full_Evaluation\((.+)\)$", stem)
    inner = match.group(1) if match else stem
    inner = inner.removesuffix(".csv")

    for judge_suffix in ["-gpt-5.2", "-gpt-5.1", "-gpt-4.1", "-gpt-4o"]:
        if inner.endswith(judge_suffix):
            return inner[: -len(judge_suffix)]

    match = re.match(r"(.+)-gpt-5\.2(?:-.*)?$", inner)
    if match:
        return match.group(1)

    return inner


def normalize_is_jailbroken(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(
        lambda value: True if value in true_values else False if value in false_values else pd.NA
    )
    return mapped.astype("boolean")


def main() -> None:
    files = sorted(ROOT_DIR.glob(FILE_PATTERN))
    if len(files) < 3:
        raise FileNotFoundError(
            f"Only {len(files)} Full_Evaluation files found. At least 3 are required."
        )

    records = []
    for file_path in files:
        df = read_csv_with_fallback(file_path)
        if "Is_Jailbroken" not in df.columns:
            raise KeyError(f"Missing 'Is_Jailbroken' column in {file_path.name}")

        jailbroken = normalize_is_jailbroken(df["Is_Jailbroken"]).dropna()
        if jailbroken.empty:
            raise ValueError(f"No valid Is_Jailbroken values found in {file_path.name}")

        success_count = int(jailbroken.sum())
        total_count = int(len(jailbroken))
        asr_percent = success_count / total_count * 100

        records.append(
            {
                "model": extract_model_name(file_path),
                "file": file_path.name,
                "success_count": success_count,
                "total_count": total_count,
                "asr_percent": asr_percent,
            }
        )

    summary = pd.DataFrame(records).sort_values("asr_percent", ascending=False)

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    palette = sns.color_palette("pastel", n_colors=len(summary))
    bars = ax.bar(
        summary["model"],
        summary["asr_percent"],
        color=palette,
        edgecolor="none",
        linewidth=0,
        width=0.65,
        alpha=0.9,
    )

    for i, (bar, value) in enumerate(zip(bars, summary["asr_percent"])):
        count_text = f"{summary.iloc[i]['success_count']}/{summary.iloc[i]['total_count']}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.4,
            f"{value:.1f}%\n({count_text})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Attack Success Rate Across Target Models", fontsize=13, pad=10)
    ax.set_xlabel("Target Model")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("ASR summary:")
    print(summary[["model", "success_count", "total_count", "asr_percent"]].to_string(index=False))
    print(f"\nSaved figure: {OUTPUT_PNG}")
    print(f"Saved table : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
