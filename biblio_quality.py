import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd


CATEGORY_ASTAR = "A*"
CATEGORY_RINC = "РИНЦ"
CATEGORY_GREY = "Серая зона"
CATEGORY_PREPRINT = "Препринт"
CATEGORY_STANDARD = "Стандарт/Спецификация"
CATEGORY_UNKNOWN = "Неизвестно"


def read_json_file(file_path: str) -> Any:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise SystemExit(f"Файл не найден: {file_path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Некорректный JSON в '{file_path}': {e}") from e


def load_bibliography(file_path: str) -> pd.DataFrame:
    data = read_json_file(file_path)
    if isinstance(data, dict):
        # Try common container keys
        for key in ("items", "records", "data", "entries"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit("Ожидался JSON-массив записей или объект с массивом по ключу 'items/records'.")
    return pd.DataFrame(data)


def get_first_present(row: pd.Series, candidates: List[str]) -> str:
    for key in candidates:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and value != "-":
                return value
    return ""


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host.replace("www.", "")
    except Exception:
        return ""


def normalize_rank(rank_raw: str) -> str:
    """Back-compat simple normalization for single-value rank fields.

    Note: richer multi-scheme parsing is in parse_rank_schemes/collect_rank_text.
    """
    rank = (rank_raw or "").strip().lower()
    # unify common variants
    replacements = {
        "а*": "a*",  # Cyrillic 'a' to Latin
        "q-1": "q1",
        "q-2": "q2",
        "q-3": "q3",
        "q-4": "q4",
        "q 1": "q1",
        "q 2": "q2",
        "q 3": "q3",
        "q 4": "q4",
        "quartile 1": "q1",
        "quartile 2": "q2",
        "quartile 3": "q3",
        "quartile 4": "q4",
    }
    for src, dst in replacements.items():
        rank = rank.replace(src, dst)
    return rank


def get_values(row: pd.Series, keys: List[str]) -> List[str]:
    values: List[str] = []
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value and value != "-":
                values.append(value)
    return values


def collect_rank_text(row: pd.Series) -> str:
    """Collect all known rank-related fields into a single text blob for parsing."""
    rank_fields = [
        # generic
        "journal_rank",
        "rank",
        "quartile",
        "scopus_quartile",
        "wos_quartile",
        "sjr_rank",
        "snip_rank",
        # business/management rankings
        "abdc_rank",
        "abdc",
        "core_rank",
        "core",
        "abs_rank",
        "ajg_rank",
        "cabs_rank",
        "ajg",
        "abs",
        "cabs",
    ]
    values = get_values(row, rank_fields)
    return " | ".join(values).lower()


def parse_rank_schemes(rank_text: str) -> Dict[str, bool]:
    """Parse rank text into boolean flags for multiple schemes.

    Returns a dict with keys like 'q1', 'q2', 'abdc_a', 'core_a*', 'abs_4*', etc.
    """
    text = (rank_text or "").lower()
    # normalize separators and dashes
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", "")
    )
    # cyrillic 'а' to latin 'a'
    text = text.replace("а*", "a*")

    flags: Dict[str, bool] = {}

    def set_flag(name: str) -> None:
        flags[name] = True

    # SJR/Scopus/WoS quartiles
    for q in ("q1", "q2", "q3", "q4"):
        if q in text or f"quartile{q[-1]}" in text:
            set_flag(q)

    # ABDC
    if any(tok in text for tok in ("abdc",)):
        if "abdca*" in text or "a*abdc" in text:
            set_flag("abdc_a*")
        for grade in ("a*", "a", "b", "c"):
            if f"abdc{grade}" in text:
                set_flag(f"abdc_{grade}")
        # also patterns like 'abdc-a', 'abdc_a'
        for grade in ("a*", "a", "b", "c"):
            if f"abdc-{grade}" in text or f"abdc_{grade}" in text:
                set_flag(f"abdc_{grade}")

    # CORE
    if any(tok in text for tok in ("core")):
        for grade in ("a*", "a", "b", "c"):
            if f"core{grade}" in text or f"core-{grade}" in text or f"core_{grade}" in text:
                set_flag(f"core_{grade}")

    # ABS/AJG/CABS (CABS/ABS/AJG levels 4*,4,3,2,1)
    for scheme in ("abs", "ajg", "cabs"):
        if scheme in text:
            for level in ("4*", "4", "3", "2", "1"):
                lvl_norm = level.replace("*", "\u2605")  # prevent accidental partial matches
                if f"{scheme}{level}" in text or f"{scheme}-{level}" in text or f"{scheme}_{level}" in text:
                    set_flag(f"abs_{level}")
                # some datasets may contain unicode star — normalize heuristically
                if f"{scheme}{lvl_norm}" in text:
                    set_flag(f"abs_{level}")

    return flags


def classify_row(row: pd.Series, treat_preprints_as_grey: bool = True) -> str:
    source = get_first_present(row, [
        "Source", "source", "Journal", "journal", "Publisher", "publisher", "Venue", "venue",
    ]).lower()
    url = get_first_present(row, ["URL", "url", "Link", "link", "Href", "href"])
    domain = extract_domain(url)
    # collect rank info across multiple possible fields
    rank_text = collect_rank_text(row)
    rank_flags = parse_rank_schemes(rank_text)

    text_blob = " ".join(filter(None, [source, domain, url])).lower()

    # RINC detection
    rinc_markers = (
        "rinc", "ринц", "elibrary", "e-library", "elibrary.ru", "e-library.ru", "cyberleninka", "cyberleninka.ru"
    )
    if any(marker in text_blob for marker in rinc_markers):
        return CATEGORY_RINC

    # Grey zone detection
    grey_markers = (
        "habr", "medium", "blog", "vc.ru", "substack", "teletype", "github.io", "dev.to", "t.me", "telegram",
        "researchgate"  # typically not peer-reviewed
    )

    # Preprint detection (optional as grey)
    preprint_markers = (
        "arxiv", "biorxiv", "bioarxiv", "medrxiv", "chemrxiv", "preprint"
    )

    if any(marker in text_blob for marker in grey_markers):
        return CATEGORY_GREY

    if any(marker in text_blob for marker in preprint_markers):
        return CATEGORY_GREY if treat_preprints_as_grey else CATEGORY_PREPRINT

    # Standards / Specifications (authoritative but not peer-reviewed)
    standard_domains = {
        "w3.org", "rfc-editor.org", "ietf.org", "iso.org", "ecma-international.org", "itu.int",
        "whatwg.org", "oasis-open.org", "khronos.org"
    }
    standard_markers = ("rfc", "recommendation", "specification", "technicalreport")
    if domain in standard_domains or any(m in text_blob for m in standard_markers):
        return CATEGORY_STANDARD

    # A* via quartiles/ranks across multiple schemes
    if (
        rank_flags.get("q1")
        or rank_flags.get("q2")
        or rank_flags.get("abdc_a*")
        or rank_flags.get("abdc_a")
        or rank_flags.get("core_a*")
        or rank_flags.get("core_a")
        or rank_flags.get("abs_4*")
        or rank_flags.get("abs_4")
    ):
        return CATEGORY_ASTAR

    return CATEGORY_UNKNOWN


def compute_stats(categories: pd.Series) -> Dict[str, float]:
    if categories.empty:
        return {}
    percentages = (categories.value_counts(normalize=True) * 100).round(2)
    return percentages.to_dict()


def print_report(stats: Dict[str, float], a_star_min: float, rinc_min: float, grey_max: float) -> None:
    print("\n📊 Отчёт по источникам:")
    if not stats:
        print("— Нет данных для отчёта")
    else:
        for cat, pct in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"— {cat}: {pct:.2f}%")

    a_star_pct = stats.get(CATEGORY_ASTAR, 0.0)
    rinc_pct = stats.get(CATEGORY_RINC, 0.0)
    grey_pct = stats.get(CATEGORY_GREY, 0.0)

    print("\n✅ Проверка критериев качества:")
    print(f"≥ {a_star_min:.0f}% — {CATEGORY_ASTAR}: ", "OK" if a_star_pct >= a_star_min else "FAIL")
    print(f"≥ {rinc_min:.0f}% — {CATEGORY_RINC}: ", "OK" if rinc_pct >= rinc_min else "FAIL")
    print(f"≤ {grey_max:.0f}% — {CATEGORY_GREY}: ", "OK" if grey_pct <= grey_max else "FAIL")


def write_excel(df: pd.DataFrame, stats: Dict[str, float], output_path: str) -> None:
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Записи")
            summary_df = (
                pd.Series(stats, name="Процент")
                .rename_axis("Категория")
                .reset_index()
            )
            summary_df.to_excel(writer, index=False, sheet_name="Сводка")
        print(f"\n💾 Подробный отчёт сохранён в '{output_path}'")
    except Exception as e:
        print(f"Не удалось записать Excel ('{output_path}'): {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка качества списка литературы (часть 3)")
    parser.add_argument("--input", required=True, help="Путь к JSON-файлу со списком литературы")
    parser.add_argument("--output", default="bibliography_report.xlsx", help="Путь к Excel-отчёту (xlsx)")
    parser.add_argument("--a-star-min", type=float, default=50.0, help="Минимальный процент A* (по умолчанию 50)")
    parser.add_argument("--rinc-min", type=float, default=15.0, help="Минимальный процент РИНЦ (по умолчанию 15)")
    parser.add_argument("--grey-max", type=float, default=10.0, help="Максимальный процент Серой зоны (по умолчанию 10)")
    parser.add_argument("--treat-preprints-as-grey", action="store_true", help="Считать arXiv/препринты серой зоной")

    args = parser.parse_args(argv)

    df = load_bibliography(args.input)
    if df.empty:
        print("Входной список пуст.")
        return 0

    df = df.copy()
    df["Категория"] = df.apply(lambda row: classify_row(row, treat_preprints_as_grey=args.treat_preprints_as_grey), axis=1)
    stats = compute_stats(df["Категория"]) or {}

    print_report(stats, a_star_min=args["a_star_min"] if isinstance(args, dict) else args.a_star_min,
                 rinc_min=args["rinc_min"] if isinstance(args, dict) else args.rinc_min,
                 grey_max=args["grey_max"] if isinstance(args, dict) else args.grey_max)

    if args.output:
        write_excel(df, stats, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
