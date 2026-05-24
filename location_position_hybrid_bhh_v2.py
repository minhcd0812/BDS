from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


RAW_TO_CANONICAL_STREET = {
    "pham dang giang": "Phạm Đăng Giang",
}

COLUMN_MAP = {
    "Phân loại kho": "phan_loai_kho",
    "Quận": "quan",
    "Phường": "phuong",
    "Phố": "pho_input",
    "duong_pho_clean": "pho",
    "Khoảng cách đến đường chính (m)": "khoang_cach_den_duong_chinh_m",
    "LT_chuan": "lt_chuan",
    "Độ rộng ngõ/ngách nhỏ nhất": "do_rong_ngo_nho_nhat_m",
    "Mục đích sử dụng đất": "muc_dich_su_dung_dat",
    "Diện tích (m2)": "dien_tich_m2",
    "Kích thước mặt tiền (m)": "mat_tien_m",
    "Kích thước chiều dài": "chieu_dai_m",
    "Số mặt tiền tiếp giáp": "so_mat_tien_tiep_giap",
    "Chi_so_hinhdang": "chi_so_hinhdang",
    "Chi_so_loithe": "chi_so_loithe",
    "Đơn giá": "don_gia",
    "Tổng giá trị": "tong_gia",
    "Năm": "nam",
    "manual_lat": "manual_lat",
    "manual_lon": "manual_lon",
    "district": "quan",
    "ward": "phuong",
    "street": "pho",
    "lat": "manual_lat",
    "lon": "manual_lon",
    "latitude": "manual_lat",
    "longitude": "manual_lon",
}

CORE_FEATURE_COLS = [
    "log_dien_tich",
    "mat_tien_m",
    "chieu_dai_m",
    "so_mat_tien_tiep_giap",
    "chi_so_hinhdang",
    "chi_so_loithe",
    "ti_le_mat_tien",
    "mat_tien_x_so_mat_tien",
]

ACCESS_FEATURE_COLS = [
    "log_khoang_cach",
    "log_do_rong_ngo",
    "lt_chuan",
    "ngo_x_lt",
]

COMPARABLE_FEATURE_COLS = [
    "total_value_band_p25",
    "total_value_band_median",
    "total_value_band_p75",
    "comparable_count",
    "comparable_confidence",
    "same_cluster_ratio",
    "access_band_match_ratio",
    "special_comparable_ratio",
    "band_confidence",
    "mean_candidate_reliability",
    "same_segment_ratio",
    "heterogeneous_comparable_ratio",
    "weighted_band_width_ratio",
]

POSITION_FEATURE_COLS = [
    "segment_lat_for_model",
    "segment_lon_for_model",
    "neighbor_count_300m",
    "neighbor_count_500m",
    "neighbor_count_1000m",
    "nearest_segment_distance_m",
]

CLUSTER_CONTEXT_FEATURE_COLS = [
    "assigned_cluster",
    "cluster_confidence",
    "cluster_sample_count",
    "distance_to_cluster_center",
]

SEGMENT_HETEROGENEITY_NUMERIC_COLS = [
    "segment_sample_count_train",
    "segment_price_cv",
    "segment_price_iqr_ratio",
    "segment_heterogeneity_score",
]


@dataclass
class PipelineConfig:
    base_dir: Path
    train_path: str = "DT_Train_binh_hung_hoa.csv"
    test_path: str = "DT_Test_Phuong_Binh_Hung_Hoa.csv"
    coord_path: str = "toa_do_Phuong_Binh_Hung_Hoa.csv"
    out_dir: str = "output_location_position_hybrid"
    artifact_dir: str = "artifacts"
    mode: str = "train_test"
    random_state: int = 42
    sigma_spatial_m: float = 700.0
    neighbor_radii_m: list[int] = field(default_factory=lambda: [300, 500, 1000, 1500, 2000])
    min_comparable_count: int = 5
    top_k_comparable: int = 5
    band_top_k: int = 12
    max_clipped_road_length_m: float = 3000.0
    spectral_dim: int = 6
    kfold_splits: int = 5
    min_cluster_size_main: int = 30
    max_cluster_candidates: int = 12
    enable_osm_evidence: bool = False
    osm_roads_path: str | None = None
    evidence_focus_row_id: str | None = None
    evidence_buffer_m: float = 1000.0
    evidence_fetch_dist_m: float = 1500.0
    evidence_map_input_limit: int = 10
    alpha: dict[str, float] = field(
        default_factory=lambda: {
            "feature": 0.35,
            "access": 0.15,
            "location": 0.20,
            "spatial": 0.30,
        }
    )

    @property
    def train_file(self) -> Path:
        return self.base_dir / self.train_path

    @property
    def test_file(self) -> Path:
        return self.base_dir / self.test_path

    @property
    def coord_file(self) -> Path:
        return self.base_dir / self.coord_path

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.out_dir

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / self.artifact_dir

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_dir"] = str(self.base_dir)
        return data


class PhaseLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        phase: str,
        objective: str,
        inference_guard: str,
        status: str,
        notes: str = "",
        outputs: list[str] | None = None,
    ) -> None:
        self.records.append(
            {
                "phase": phase,
                "objective": objective,
                "inference_guard": inference_guard,
                "status": status,
                "notes": notes,
                "outputs": outputs or [],
            }
        )

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        output_1_dir = out_dir / "output_1"
        output_1_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_1_dir / "workflow_phase_log.json"
        md_path = output_1_dir / "workflow_phase_log.md"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.records, handle, ensure_ascii=False, indent=2)
        lines = ["# Workflow Phase Log", ""]
        for item in self.records:
            lines.append(f"## {item['phase']}")
            lines.append(f"- Objective: {item['objective']}")
            lines.append(f"- Inference guard: {item['inference_guard']}")
            lines.append(f"- Status: {item['status']}")
            if item["notes"]:
                lines.append(f"- Notes: {item['notes']}")
            if item["outputs"]:
                lines.append(f"- Outputs: {', '.join(item['outputs'])}")
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("YAML config requires PyYAML to be installed.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Location + Position Hybrid Workflow for Binh Hung Hoa")
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-dir", default="/Users/chudinhminh/Downloads/E")
    parser.add_argument("--train-path", default="DT_Train_binh_hung_hoa.csv")
    parser.add_argument("--test-path", default="DT_Test_Phuong_Binh_Hung_Hoa.csv")
    parser.add_argument("--coord-path", default="toa_do_Phuong_Binh_Hung_Hoa.csv")
    parser.add_argument("--out-dir", default="output_location_position_hybrid")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sigma-spatial-m", type=float, default=700.0)
    parser.add_argument("--neighbor-radii-m", nargs="*", type=int, default=[300, 500, 1000, 1500, 2000])
    parser.add_argument("--min-comparable-count", type=int, default=5)
    parser.add_argument("--top-k-comparable", type=int, default=5)
    parser.add_argument("--band-top-k", type=int, default=12)
    parser.add_argument("--max-clipped-road-length-m", type=float, default=3000.0)
    parser.add_argument("--spectral-dim", type=int, default=6)
    parser.add_argument("--kfold-splits", type=int, default=5)
    parser.add_argument("--min-cluster-size-main", type=int, default=30)
    parser.add_argument("--max-cluster-candidates", type=int, default=12)
    parser.add_argument("--enable-osm-evidence", action="store_true")
    parser.add_argument("--osm-roads-path", default=None)
    parser.add_argument("--evidence-focus-row-id", default=None)
    parser.add_argument("--evidence-buffer-m", type=float, default=1000.0)
    parser.add_argument("--evidence-fetch-dist-m", type=float, default=1500.0)
    parser.add_argument("--evidence-map-input-limit", type=int, default=10)
    args = parser.parse_args()

    base = PipelineConfig(base_dir=Path(args.base_dir))
    config_data = base.to_json_dict()
    if args.config:
        config_data.update(load_config_file(Path(args.config)))

    cli_overrides = {
        "base_dir": args.base_dir,
        "train_path": args.train_path,
        "test_path": args.test_path,
        "coord_path": args.coord_path,
        "out_dir": args.out_dir,
        "artifact_dir": args.artifact_dir,
        "mode": args.mode,
        "random_state": args.random_state,
        "sigma_spatial_m": args.sigma_spatial_m,
        "neighbor_radii_m": args.neighbor_radii_m,
        "min_comparable_count": args.min_comparable_count,
        "top_k_comparable": args.top_k_comparable,
        "band_top_k": args.band_top_k,
        "max_clipped_road_length_m": args.max_clipped_road_length_m,
        "spectral_dim": args.spectral_dim,
        "kfold_splits": args.kfold_splits,
        "min_cluster_size_main": args.min_cluster_size_main,
        "max_cluster_candidates": args.max_cluster_candidates,
        "enable_osm_evidence": args.enable_osm_evidence,
        "osm_roads_path": args.osm_roads_path,
        "evidence_focus_row_id": args.evidence_focus_row_id,
        "evidence_buffer_m": args.evidence_buffer_m,
        "evidence_fetch_dist_m": args.evidence_fetch_dist_m,
        "evidence_map_input_limit": args.evidence_map_input_limit,
    }
    config_data.update(cli_overrides)
    config_data["base_dir"] = Path(config_data["base_dir"])
    return PipelineConfig(**config_data)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in df.columns:
        clean = str(column).replace("\ufeff", "").strip()
        renamed[column] = COLUMN_MAP.get(clean, clean)
    out = df.rename(columns=renamed).copy()
    duplicate_names = [name for name in out.columns if list(out.columns).count(name) > 1]
    if duplicate_names:
        unique_names = list(dict.fromkeys(duplicate_names))
        for name in unique_names:
            dup = out.loc[:, out.columns == name]
            merged = dup.bfill(axis=1).iloc[:, 0]
            out = out.loc[:, out.columns != name]
            out[name] = merged
    return out


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.lower().replace("đ", "d").replace("Đ", "D")
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = re.sub(r"\bq\.\s*", "quan ", text)
    text = re.sub(r"\bq\s+", "quan ", text)
    text = re.sub(r"\bp\.\s*", "phuong ", text)
    text = re.sub(r"\bp\s+", "phuong ", text)
    text = re.sub(r"[^0-9a-z/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text == "pham dang giang":
        return "pham dang giang"
    return text


def normalize_admin(value: object, prefix: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    if not text.startswith(f"{prefix} "):
        text = f"{prefix} {text}"
    return re.sub(r"\s+", " ", text).strip()


def make_segment_key(quan: object, phuong: object, pho: object) -> str:
    return "|".join(
        [
            normalize_admin(quan, "quan"),
            normalize_admin(phuong, "phuong"),
            normalize_text(pho),
        ]
    )


def canonical_street_display(raw_value: object, canonical_lookup: dict[str, str]) -> str:
    normalized = normalize_text(raw_value)
    if not normalized:
        return ""
    if normalized in RAW_TO_CANONICAL_STREET:
        return RAW_TO_CANONICAL_STREET[normalized]
    return canonical_lookup.get(normalized, str(raw_value).strip())


def join_flags(values: Iterable[str]) -> str:
    clean = [value for value in values if value and value != "OK"]
    clean = list(dict.fromkeys(clean))
    return "OK" if not clean else "|".join(clean)


def parse_manual_coordinate(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return np.nan
    if abs(numeric) > 1000:
        numeric = numeric / 1_000_000
    return float(numeric)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def safe_mape(actual: pd.Series, pred: pd.Series) -> float:
    mask = actual.notna() & pred.notna() & (actual != 0)
    if not mask.any():
        return np.nan
    return float((np.abs(actual[mask] - pred[mask]) / actual[mask]).mean())


def mean_abs_log_error(actual: pd.Series, pred: pd.Series) -> float:
    mask = actual.notna() & pred.notna() & (actual > 0) & (pred > 0)
    if not mask.any():
        return np.nan
    return float(np.abs(np.log1p(actual[mask]) - np.log1p(pred[mask])).mean())


def format_money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> list[str]:
    missing = [column for column in required_cols if column not in df.columns]
    return missing


def build_input_schema_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    coord_df: pd.DataFrame,
) -> dict[str, Any]:
    train_required = [
        "quan",
        "phuong",
        "pho",
        "tong_gia",
        "dien_tich_m2",
        "mat_tien_m",
        "chieu_dai_m",
        "so_mat_tien_tiep_giap",
        "khoang_cach_den_duong_chinh_m",
        "do_rong_ngo_nho_nhat_m",
        "lt_chuan",
    ]
    test_required = [
        "quan",
        "phuong",
        "pho",
        "dien_tich_m2",
        "mat_tien_m",
        "chieu_dai_m",
        "so_mat_tien_tiep_giap",
        "khoang_cach_den_duong_chinh_m",
        "do_rong_ngo_nho_nhat_m",
        "lt_chuan",
    ]
    coord_required = ["quan", "phuong", "pho", "manual_lat", "manual_lon"]

    train_missing = validate_required_columns(train_df, train_required, "train")
    test_missing = validate_required_columns(test_df, test_required, "test")
    coord_missing = validate_required_columns(coord_df, coord_required, "coord")
    test_has_target = "tong_gia" in test_df.columns and test_df["tong_gia"].notna().any()
    schema_status = "OK" if not train_missing and not test_missing and not coord_missing else "ERROR"
    return {
        "train_missing_columns": train_missing,
        "test_missing_columns": test_missing,
        "coord_missing_columns": coord_missing,
        "test_has_target": bool(test_has_target),
        "schema_status": schema_status,
    }


def read_csv_file(path: Path) -> pd.DataFrame:
    return clean_headers(pd.read_csv(path))


def build_canonical_lookup(*dfs: pd.DataFrame) -> dict[str, str]:
    frames = []
    for df in dfs:
        temp = df.copy()
        temp["raw_street"] = temp["pho"].fillna("").astype(str).str.strip()
        temp["normalized_street"] = temp["raw_street"].map(normalize_text)
        frames.append(temp[["raw_street", "normalized_street"]])
    base = pd.concat(frames, ignore_index=True)
    lookup: dict[str, str] = {}
    for normalized, group in base.groupby("normalized_street"):
        if not normalized:
            lookup[normalized] = ""
            continue
        if normalized in RAW_TO_CANONICAL_STREET:
            lookup[normalized] = RAW_TO_CANONICAL_STREET[normalized]
            continue
        raw_counts = group.loc[group["raw_street"] != "", "raw_street"].value_counts()
        lookup[normalized] = raw_counts.idxmax()
    return lookup


def build_alias_report(canonical_lookup: dict[str, str], *dfs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for df in dfs:
        temp = df.copy()
        temp["raw_street"] = temp["pho"].fillna("").astype(str).str.strip()
        temp["normalized_street"] = temp["raw_street"].map(normalize_text)
        frames.append(temp[["raw_street", "normalized_street"]])
    base = pd.concat(frames, ignore_index=True)
    if base.empty:
        return pd.DataFrame(
            columns=["raw_street", "normalized_street", "canonical_street", "alias_status", "row_count"]
        )
    distinct_counts = base.groupby("normalized_street")["raw_street"].nunique()
    report = (
        base.groupby(["raw_street", "normalized_street"])
        .size()
        .reset_index(name="row_count")
        .sort_values(["normalized_street", "row_count", "raw_street"], ascending=[True, False, True])
    )
    report["canonical_street"] = report["normalized_street"].map(canonical_lookup)
    report["alias_status"] = np.where(
        report["normalized_street"] == "",
        "MISSING_STREET_NAME",
        np.where(
            report["raw_street"] != report["canonical_street"],
            "ALIAS_MERGED",
            np.where(
                report["normalized_street"].map(distinct_counts).fillna(1) > 1,
                "CANONICAL_WITH_ALIAS_GROUP",
                "CANONICAL",
            ),
        ),
    )
    return report[
        ["raw_street", "normalized_street", "canonical_street", "alias_status", "row_count"]
    ].reset_index(drop=True)


def add_basic_columns(df: pd.DataFrame, dataset_name: str, canonical_lookup: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["dataset_name"] = dataset_name
    out["row_id"] = [f"{dataset_name}_{i:04d}" for i in range(1, len(out) + 1)]
    if "pho_input" not in out.columns:
        out["pho_input"] = out.get("pho", "")
    out["pho_raw"] = out.get("pho", "").fillna("").astype(str).str.strip()
    out["pho_norm"] = out["pho_raw"].map(normalize_text)
    out["pho_canonical"] = out["pho_raw"].map(lambda value: canonical_street_display(value, canonical_lookup))
    out["pho_canonical_norm"] = out["pho_canonical"].map(normalize_text)
    out["quan_norm"] = out["quan"].map(lambda value: normalize_admin(value, "quan"))
    out["phuong_norm"] = out["phuong"].map(lambda value: normalize_admin(value, "phuong"))
    out["segment_key"] = out.apply(
        lambda row: make_segment_key(row["quan"], row["phuong"], row["pho_canonical"]),
        axis=1,
    )
    out["row_input_lat"] = (
        out["manual_lat"].map(parse_manual_coordinate) if "manual_lat" in out.columns else np.nan
    )
    out["row_input_lon"] = (
        out["manual_lon"].map(parse_manual_coordinate) if "manual_lon" in out.columns else np.nan
    )
    return out


def prepare_coordinate_dictionary(coord_df: pd.DataFrame, canonical_lookup: dict[str, str]) -> pd.DataFrame:
    coord = add_basic_columns(coord_df, "coord", canonical_lookup)
    coord["segment_lat"] = coord["row_input_lat"]
    coord["segment_lon"] = coord["row_input_lon"]
    coord["coord_source"] = "manual_coordinate_dictionary"
    return coord


def build_segment_master(train_df: pd.DataFrame, coord_df: pd.DataFrame) -> pd.DataFrame:
    train_counts = (
        train_df.groupby("segment_key")
        .agg(
            train_sample_count=("row_id", "size"),
            quan=("quan", "first"),
            phuong=("phuong", "first"),
            pho=("pho_canonical", "first"),
        )
        .reset_index()
    )
    coord_unique = coord_df[
        [
            "segment_key",
            "quan",
            "phuong",
            "pho_canonical",
            "segment_lat",
            "segment_lon",
            "coord_source",
        ]
    ].drop_duplicates("segment_key")
    coord_unique = coord_unique.rename(columns={"pho_canonical": "pho"})

    segment_master = pd.merge(
        coord_unique,
        train_counts,
        on="segment_key",
        how="outer",
        suffixes=("_coord", "_train"),
    )
    segment_master["quan"] = segment_master["quan_coord"].fillna(segment_master["quan_train"])
    segment_master["phuong"] = segment_master["phuong_coord"].fillna(segment_master["phuong_train"])
    segment_master["pho"] = segment_master["pho_coord"].fillna(segment_master["pho_train"]).fillna("")
    segment_master["train_sample_count"] = segment_master["train_sample_count"].fillna(0).astype(int)
    segment_master["sample_count"] = segment_master["train_sample_count"]
    segment_master["coord_found"] = segment_master["segment_lat"].notna() & segment_master["segment_lon"].notna()
    segment_master["coord_source"] = np.where(
        segment_master["coord_found"],
        segment_master["coord_source"].fillna("manual_coordinate_dictionary"),
        "unmatched",
    )
    segment_master = segment_master.drop(
        columns=["quan_coord", "phuong_coord", "pho_coord", "quan_train", "phuong_train", "pho_train"]
    )

    valid_range = segment_master["segment_lat"].between(10.0, 11.5) & segment_master["segment_lon"].between(
        106.0, 107.2
    )
    ward_median_lat = segment_master.loc[valid_range, "segment_lat"].median()
    ward_median_lon = segment_master.loc[valid_range, "segment_lon"].median()
    segment_master["distance_to_ward_median_m"] = segment_master.apply(
        lambda row: haversine_m(row["segment_lat"], row["segment_lon"], ward_median_lat, ward_median_lon),
        axis=1,
    )
    duplicate_counts = (
        segment_master.loc[segment_master["coord_found"]]
        .groupby(["segment_lat", "segment_lon"])
        .size()
        .to_dict()
    )

    q1 = segment_master["distance_to_ward_median_m"].quantile(0.25)
    q3 = segment_master["distance_to_ward_median_m"].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = max(float(q3 + 4 * iqr), 5000.0)

    flags = []
    notes = []
    coord_match_status = []
    lat_for_model = []
    lon_for_model = []
    for row in segment_master.itertuples(index=False):
        row_flags: list[str] = []
        note = "Coordinate parsed from manual dictionary."
        if not row.coord_found:
            row_flags.append("MISSING_SEGMENT_GEO")
            note = "Street name is blank or missing in the coordinate dictionary, so no coordinate match is available."
        else:
            if not (10.0 <= row.segment_lat <= 11.5 and 106.0 <= row.segment_lon <= 107.2):
                row_flags.extend(["LAT_LON_OUT_OF_RANGE", "COORD_NEEDS_MANUAL_REVIEW"])
                note = "Coordinate is outside the expected Ho Chi Minh City range."
            if duplicate_counts.get((row.segment_lat, row.segment_lon), 0) > 1:
                row_flags.append("DUPLICATED_COORDINATE")
            if pd.notna(row.distance_to_ward_median_m) and row.distance_to_ward_median_m > outlier_threshold:
                row_flags.extend(["WARD_COORD_OUTLIER", "COORD_NEEDS_MANUAL_REVIEW"])
                note = "Coordinate is far from the ward coordinate median; manual review is recommended."
            if normalize_text(row.pho) == "duong so 12" and pd.notna(row.distance_to_ward_median_m):
                if row.distance_to_ward_median_m > 10000:
                    if "WARD_COORD_OUTLIER" not in row_flags:
                        row_flags.append("WARD_COORD_OUTLIER")
                    if "COORD_NEEDS_MANUAL_REVIEW" not in row_flags:
                        row_flags.append("COORD_NEEDS_MANUAL_REVIEW")
                    note = (
                        "Đường Số 12 has longitude 106.766142 and is far from the Bình Hưng Hòa cluster; "
                        "manual verification is required."
                    )
        warning = join_flags(row_flags)
        flags.append(warning)
        notes.append(note)
        coord_match_status.append("MATCHED_MANUAL_COORD" if row.coord_found else "MISSING_SEGMENT_GEO")
        if any(flag in row_flags for flag in ["MISSING_SEGMENT_GEO", "LAT_LON_OUT_OF_RANGE", "WARD_COORD_OUTLIER"]):
            lat_for_model.append(np.nan)
            lon_for_model.append(np.nan)
        else:
            lat_for_model.append(row.segment_lat)
            lon_for_model.append(row.segment_lon)

    segment_master["coord_warning_flag"] = flags
    segment_master["geo_note"] = notes
    segment_master["coord_match_status"] = coord_match_status
    segment_master["segment_lat_for_model"] = lat_for_model
    segment_master["segment_lon_for_model"] = lon_for_model
    return segment_master[
        [
            "segment_key",
            "quan",
            "phuong",
            "pho",
            "sample_count",
            "train_sample_count",
            "segment_lat",
            "segment_lon",
            "coord_source",
            "coord_found",
            "coord_match_status",
            "coord_warning_flag",
            "distance_to_ward_median_m",
            "geo_note",
            "segment_lat_for_model",
            "segment_lon_for_model",
        ]
    ].copy()


def attach_segment_info(df: pd.DataFrame, segment_master: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = df.merge(segment_master, on="segment_key", how="left", suffixes=("", "_segment"))
    out["segment_lat"] = out["segment_lat"].combine_first(out["row_input_lat"])
    out["segment_lon"] = out["segment_lon"].combine_first(out["row_input_lon"])
    out["segment_lat_for_model"] = out["segment_lat_for_model"].combine_first(
        out["row_input_lat"].where(out["row_input_lat"].between(10.0, 11.5))
    )
    out["segment_lon_for_model"] = out["segment_lon_for_model"].combine_first(
        out["row_input_lon"].where(out["row_input_lon"].between(106.0, 107.2))
    )
    out["has_valid_coordinate"] = out["segment_lat_for_model"].notna() & out["segment_lon_for_model"].notna()
    out["location_warning_flag"] = out.apply(
        lambda row: join_flags(
            [
                "MISSING_STREET_NAME" if row["pho_norm"] == "" else "",
                "STREET_ALIAS_MERGED"
                if row["pho_raw"] and row["pho_canonical"] and row["pho_raw"] != row["pho_canonical"]
                else "",
            ]
        ),
        axis=1,
    )
    if is_train:
        out["target_warning_flag"] = np.where(out["tong_gia"].fillna(0) <= 0, "ZERO_TOTAL_VALUE_REMOVED", "OK")
    else:
        out["target_warning_flag"] = "OK"
    out["log_tong_gia"] = np.where(out.get("tong_gia", pd.Series(index=out.index)).fillna(0) > 0, np.log1p(out["tong_gia"]), np.nan)
    out["warning_flag_base"] = out.apply(
        lambda row: join_flags(
            [row.get("coord_warning_flag", "OK"), row["location_warning_flag"], row["target_warning_flag"]]
        ),
        axis=1,
    )
    return out


def build_frequency_lookup(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "segment_count": train_df.groupby("segment_key").size().astype(int).to_dict(),
        "phuong_count": train_df.groupby("phuong_norm").size().astype(int).to_dict(),
        "quan_count": train_df.groupby("quan_norm").size().astype(int).to_dict(),
    }


def add_engineered_features(df: pd.DataFrame, frequency_lookup: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    lt_chuan = pd.to_numeric(out["lt_chuan"], errors="coerce")
    ngo_width = pd.to_numeric(out["do_rong_ngo_nho_nhat_m"], errors="coerce")
    main_road_distance = pd.to_numeric(out["khoang_cach_den_duong_chinh_m"], errors="coerce")
    out["log_dien_tich"] = np.log1p(pd.to_numeric(out["dien_tich_m2"], errors="coerce").clip(lower=0))
    out["log_khoang_cach"] = np.log1p(main_road_distance.clip(lower=0))
    out["log_do_rong_ngo"] = np.log1p(ngo_width.clip(lower=0))
    out["ti_le_mat_tien"] = out["mat_tien_m"] / out["chieu_dai_m"].replace(0, np.nan)
    out["mat_tien_x_so_mat_tien"] = out["mat_tien_m"] * out["so_mat_tien_tiep_giap"]
    out["ngo_x_lt"] = out["do_rong_ngo_nho_nhat_m"] * out["lt_chuan"]
    out["access_band"] = np.select(
        [
            (lt_chuan >= 4) | ((ngo_width >= 8) & (main_road_distance <= 60)),
            (lt_chuan >= 3) | (ngo_width >= 6) | (main_road_distance <= 30),
            (lt_chuan >= 2) | (ngo_width >= 4),
        ],
        [
            "mat_tien",
            "hem_lon",
            "hem_trung_binh",
        ],
        default="hem_nho_sau",
    )
    out["access_score"] = np.clip(
        0.50 * ((lt_chuan.fillna(1) - 1) / 3.0)
        + 0.30 * np.clip(ngo_width.fillna(0) / 8.0, 0, 1)
        + 0.20 * (1 - np.clip(main_road_distance.fillna(300) / 250.0, 0, 1)),
        0,
        1,
    )
    out["pho_count_train"] = out["segment_key"].map(frequency_lookup["segment_count"]).fillna(0).astype(int)
    out["phuong_count_train"] = out["phuong_norm"].map(frequency_lookup["phuong_count"]).fillna(0).astype(int)
    out["pho_rare"] = (out["pho_count_train"] < 15).astype(int)
    out["phuong_rare"] = (out["phuong_count_train"] < 10).astype(int)
    return out


def weighted_quantile(values: Iterable[float], quantile: float, sample_weight: Iterable[float]) -> float:
    value_arr = np.asarray(list(values), dtype=float)
    weight_arr = np.asarray(list(sample_weight), dtype=float)
    valid_mask = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not valid_mask.any():
        return np.nan
    value_arr = value_arr[valid_mask]
    weight_arr = weight_arr[valid_mask]
    order = np.argsort(value_arr)
    value_arr = value_arr[order]
    weight_arr = weight_arr[order]
    cumulative = np.cumsum(weight_arr) - 0.5 * weight_arr
    cumulative = cumulative / max(weight_arr.sum(), 1e-6)
    return float(np.interp(np.clip(quantile, 0, 1), cumulative, value_arr))


def build_segment_heterogeneity_table(clean_train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = clean_train_df.copy()
    unit_price = pd.to_numeric(train.get("don_gia"), errors="coerce")
    if unit_price.notna().sum() == 0:
        unit_price = pd.to_numeric(train["tong_gia"], errors="coerce") / pd.to_numeric(
            train["dien_tich_m2"], errors="coerce"
        ).replace(0, np.nan)
    train["unit_price_for_diag"] = unit_price
    segment_stats = (
        train.groupby("segment_key")
        .agg(
            segment_sample_count_train=("row_id", "size"),
            segment_unit_price_mean=("unit_price_for_diag", "mean"),
            segment_unit_price_median=("unit_price_for_diag", "median"),
            segment_unit_price_std=("unit_price_for_diag", "std"),
            segment_unit_price_p25=("unit_price_for_diag", lambda series: float(series.quantile(0.25))),
            segment_unit_price_p75=("unit_price_for_diag", lambda series: float(series.quantile(0.75))),
            phuong_norm=("phuong_norm", "first"),
            quan_norm=("quan_norm", "first"),
        )
        .reset_index()
    )
    segment_stats["segment_price_cv"] = (
        segment_stats["segment_unit_price_std"] / np.maximum(segment_stats["segment_unit_price_mean"], 1e-6)
    )
    segment_stats["segment_price_iqr_ratio"] = (
        segment_stats["segment_unit_price_p75"] - segment_stats["segment_unit_price_p25"]
    ) / np.maximum(segment_stats["segment_unit_price_median"], 1e-6)

    valid_cv = segment_stats[
        (segment_stats["segment_sample_count_train"] >= 3)
        & segment_stats["segment_price_cv"].notna()
        & np.isfinite(segment_stats["segment_price_cv"])
    ]["segment_price_cv"]
    valid_iqr = segment_stats[
        (segment_stats["segment_sample_count_train"] >= 3)
        & segment_stats["segment_price_iqr_ratio"].notna()
        & np.isfinite(segment_stats["segment_price_iqr_ratio"])
    ]["segment_price_iqr_ratio"]

    cv_p75 = float(valid_cv.quantile(0.75)) if not valid_cv.empty else 0.25
    cv_p90 = float(valid_cv.quantile(0.90)) if not valid_cv.empty else max(cv_p75, 0.35)
    iqr_p75 = float(valid_iqr.quantile(0.75)) if not valid_iqr.empty else 0.35
    iqr_p90 = float(valid_iqr.quantile(0.90)) if not valid_iqr.empty else max(iqr_p75, 0.50)

    cv_scale = max(cv_p90, 1e-6)
    iqr_scale = max(iqr_p90, 1e-6)
    segment_stats["segment_heterogeneity_score"] = np.clip(
        0.60 * (segment_stats["segment_price_cv"].fillna(cv_p75) / cv_scale)
        + 0.40 * (segment_stats["segment_price_iqr_ratio"].fillna(iqr_p75) / iqr_scale),
        0,
        2.0,
    )
    segment_stats["segment_heterogeneity_flag"] = np.select(
        [
            segment_stats["segment_sample_count_train"] < 3,
            (segment_stats["segment_price_cv"] > cv_p90) | (segment_stats["segment_price_iqr_ratio"] > iqr_p90),
            (segment_stats["segment_price_cv"] > cv_p75) | (segment_stats["segment_price_iqr_ratio"] > iqr_p75),
        ],
        [
            "LOW_SAMPLE_SEGMENT",
            "HETEROGENEOUS_HIGH",
            "HETEROGENEOUS_MEDIUM",
        ],
        default="STABLE_SEGMENT",
    )
    segment_stats["segment_heterogeneity_penalty"] = segment_stats["segment_heterogeneity_flag"].map(
        {
            "STABLE_SEGMENT": 1.00,
            "HETEROGENEOUS_MEDIUM": 0.88,
            "HETEROGENEOUS_HIGH": 0.75,
            "LOW_SAMPLE_SEGMENT": 0.82,
        }
    ).fillna(0.85)

    config = {
        "cv_p75": cv_p75,
        "cv_p90": cv_p90,
        "iqr_p75": iqr_p75,
        "iqr_p90": iqr_p90,
        "global_numeric_defaults": {
            "segment_sample_count_train": float(segment_stats["segment_sample_count_train"].median()),
            "segment_price_cv": float(segment_stats["segment_price_cv"].fillna(cv_p75).median()),
            "segment_price_iqr_ratio": float(segment_stats["segment_price_iqr_ratio"].fillna(iqr_p75).median()),
            "segment_heterogeneity_score": float(segment_stats["segment_heterogeneity_score"].median()),
            "segment_heterogeneity_penalty": float(segment_stats["segment_heterogeneity_penalty"].median()),
        },
        "global_flag_default": str(segment_stats["segment_heterogeneity_flag"].mode().iloc[0]),
    }
    return segment_stats, config


def build_segment_heterogeneity_templates(
    segment_heterogeneity_df: pd.DataFrame,
    heterogeneity_config: dict[str, Any],
) -> dict[str, Any]:
    numeric_cols = SEGMENT_HETEROGENEITY_NUMERIC_COLS + ["segment_heterogeneity_penalty"]
    templates: dict[str, Any] = {"global": {}, "phuong": {}, "quan": {}}
    for column, value in heterogeneity_config["global_numeric_defaults"].items():
        templates["global"][column] = float(value)
    templates["global"]["segment_heterogeneity_flag"] = heterogeneity_config["global_flag_default"]

    for level in ["phuong_norm", "quan_norm"]:
        numeric_group = segment_heterogeneity_df.groupby(level)[numeric_cols].mean().reset_index()
        flag_group = (
            segment_heterogeneity_df.groupby(level)["segment_heterogeneity_flag"]
            .agg(lambda series: str(pd.Series(series).mode().iloc[0]))
            .reset_index()
        )
        merged = numeric_group.merge(flag_group, on=level, how="left")
        target_key = "phuong" if level == "phuong_norm" else "quan"
        templates[target_key] = merged.set_index(level).to_dict(orient="index")
    return templates


def attach_segment_heterogeneity(
    df: pd.DataFrame,
    segment_heterogeneity_df: pd.DataFrame,
    heterogeneity_templates: dict[str, Any],
) -> pd.DataFrame:
    out = df.merge(
        segment_heterogeneity_df[
            [
                "segment_key",
                *SEGMENT_HETEROGENEITY_NUMERIC_COLS,
                "segment_heterogeneity_flag",
                "segment_heterogeneity_penalty",
            ]
        ],
        on="segment_key",
        how="left",
    )
    source_values = []
    for idx, row in out.iterrows():
        if pd.notna(row.get("segment_heterogeneity_score")):
            source_values.append("SEGMENT_LOOKUP")
            continue
        phuong_template = heterogeneity_templates["phuong"].get(row["phuong_norm"])
        quan_template = heterogeneity_templates["quan"].get(row["quan_norm"])
        template = phuong_template or quan_template or heterogeneity_templates["global"]
        for column in SEGMENT_HETEROGENEITY_NUMERIC_COLS + ["segment_heterogeneity_penalty"]:
            out.at[idx, column] = float(template.get(column, heterogeneity_templates["global"].get(column, 0.0)))
        out.at[idx, "segment_heterogeneity_flag"] = template.get(
            "segment_heterogeneity_flag",
            heterogeneity_templates["global"]["segment_heterogeneity_flag"],
        )
        source_values.append("PHUONG_TEMPLATE" if phuong_template else "QUAN_TEMPLATE" if quan_template else "GLOBAL_TEMPLATE")
    out["segment_heterogeneity_source"] = source_values
    out["segment_heterogeneity_penalty"] = out["segment_heterogeneity_penalty"].fillna(0.85)
    out["segment_heterogeneity_flag"] = out["segment_heterogeneity_flag"].fillna("UNKNOWN_SEGMENT")
    return out


def fit_location_artifacts(train_df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = train_df.copy()
    artifacts: dict[str, Any] = {
        "encoders": {},
        "svds": {},
        "scalers": {},
        "feature_names": [],
        "specs": [
            ("quan_norm", "loc_quan", 2),
            ("phuong_norm", "loc_phuong", 4),
            ("pho_canonical_norm", "loc_pho", 16),
        ],
    }
    for source_col, prefix, max_components in artifacts["specs"]:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        train_matrix = encoder.fit_transform(train[[source_col]])
        n_features = train_matrix.shape[1]
        if n_features <= 1:
            component_names = [f"{prefix}_1"]
            embedding = np.zeros((len(train), 1))
            svd = None
            scaler = None
        else:
            n_components = max(1, min(max_components, n_features - 1, len(train) - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            scaler = RobustScaler()
            embedding = scaler.fit_transform(svd.fit_transform(train_matrix))
            component_names = [f"{prefix}_{i + 1}" for i in range(n_components)]
        for idx, name in enumerate(component_names):
            train[name] = embedding[:, idx]
        artifacts["encoders"][source_col] = encoder
        artifacts["svds"][source_col] = svd
        artifacts["scalers"][source_col] = scaler
        artifacts["feature_names"].extend(component_names)
    return train, artifacts


def transform_location_features(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for source_col, prefix, _ in artifacts["specs"]:
        encoder = artifacts["encoders"][source_col]
        svd = artifacts["svds"][source_col]
        scaler = artifacts["scalers"][source_col]
        matrix = encoder.transform(out[[source_col]])
        if svd is None:
            embedding = np.zeros((len(out), 1))
            component_names = [f"{prefix}_1"]
        else:
            transformed = svd.transform(matrix)
            embedding = scaler.transform(transformed)
            component_names = [f"{prefix}_{i + 1}" for i in range(embedding.shape[1])]
        for idx, name in enumerate(component_names):
            out[name] = embedding[:, idx]
    for name in artifacts["feature_names"]:
        if name not in out.columns:
            out[name] = 0.0
    return out


def fit_scaled_bundle(train_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    bundle = {
        "columns": columns,
        "imputer": SimpleImputer(strategy="median"),
        "scaler": RobustScaler(),
    }
    values = bundle["imputer"].fit_transform(train_df[columns])
    bundle["scaler"].fit(values)
    return bundle


def transform_scaled_bundle(df: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    values = bundle["imputer"].transform(df[bundle["columns"]])
    return bundle["scaler"].transform(values)


def normalize_affinity_graph(graph: sparse.csr_matrix) -> sparse.csr_matrix:
    if graph.nnz == 0:
        return graph.tocsr()
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(row_sums, 1e-6))
    degree = sparse.diags(inv_sqrt)
    normalized = (degree @ graph @ degree).tocsr()
    normalized.eliminate_zeros()
    return normalized


def boost_graph_for_matching_context(
    graph: sparse.csr_matrix,
    labels: Iterable[object],
    boost_if_same: float,
) -> sparse.csr_matrix:
    if graph.nnz == 0:
        return graph.tocsr()
    out = graph.tocsr(copy=True)
    label_array = np.asarray(list(labels), dtype=object)
    row_idx, col_idx = out.nonzero()
    same_mask = label_array[row_idx] == label_array[col_idx]
    out.data = out.data * np.where(same_mask, boost_if_same, 1.0)
    return ((out + out.T) * 0.5).tocsr()


def graph_local_density(graph: sparse.csr_matrix, top_k: int = 8) -> np.ndarray:
    density = np.zeros(graph.shape[0], dtype=float)
    csr = graph.tocsr()
    for idx in range(csr.shape[0]):
        start, end = csr.indptr[idx], csr.indptr[idx + 1]
        values = csr.data[start:end]
        positive = values[values > 0]
        if positive.size == 0:
            continue
        density[idx] = float(np.mean(np.sort(positive)[-top_k:]))
    return np.clip(density, 0, 1)


def cluster_assignment_details(
    cluster_model: KMeans,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusters = cluster_model.predict(embedding).astype(int)
    distances = cluster_model.transform(embedding)
    assigned_distance = distances[np.arange(len(embedding)), clusters]
    if distances.shape[1] > 1:
        second_best = np.partition(distances, 1, axis=1)[:, 1]
    else:
        second_best = assigned_distance + 1.0
    confidence = 1 - (assigned_distance / np.maximum(second_best, 1e-6))
    return clusters, np.clip(confidence, 0, 1), assigned_distance


def choose_cluster_count(
    embedding: np.ndarray,
    train_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[int, pd.DataFrame]:
    if len(train_df) <= 4:
        report = pd.DataFrame(
            [
                {
                    "n_clusters": 2,
                    "silhouette_score": np.nan,
                    "price_separation_score": np.nan,
                    "mean_cluster_confidence": np.nan,
                    "min_cluster_size": len(train_df),
                    "median_cluster_size": len(train_df),
                    "max_cluster_size": len(train_df),
                    "dominant_segment_share_mean": 1.0,
                    "cluster_score": 0.0,
                    "selected": True,
                }
            ]
        )
        return 2, report

    upper_bound = min(
        max(3, train_df["segment_key"].nunique()),
        config.max_cluster_candidates,
        max(4, int(math.sqrt(max(len(train_df), 1) / 18.0)) + 2),
    )
    lower_bound = min(3, upper_bound)
    candidate_range = range(lower_bound, upper_bound + 1)
    positive_target = train_df["tong_gia"] > 0
    candidate_rows: list[dict[str, Any]] = []

    for n_clusters in candidate_range:
        cluster_model = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=30)
        clusters = cluster_model.fit_predict(embedding)
        counts = pd.Series(clusters).value_counts().sort_index()
        mean_confidence = np.nan
        silhouette = np.nan
        price_score = np.nan
        dominant_segment_share_mean = np.nan

        if len(np.unique(clusters)) > 1:
            silhouette = float(silhouette_score(embedding, clusters))
        pred_clusters, confidence, _ = cluster_assignment_details(cluster_model, embedding)
        mean_confidence = float(np.mean(confidence))

        cluster_segment = pd.DataFrame(
            {
                "cluster": pred_clusters,
                "segment_key": train_df["segment_key"].to_numpy(),
            }
        )
        dominant_segment_share = (
            cluster_segment.groupby("cluster")["segment_key"]
            .value_counts(normalize=True)
            .groupby(level=0)
            .max()
        )
        dominant_segment_share_mean = float(dominant_segment_share.mean()) if not dominant_segment_share.empty else 1.0

        if positive_target.sum() > n_clusters:
            price_eval = pd.DataFrame(
                {
                    "cluster": pred_clusters[positive_target.to_numpy()],
                    "log_target": np.log1p(train_df.loc[positive_target, "tong_gia"].to_numpy()),
                }
            )
            cluster_median = price_eval.groupby("cluster")["log_target"].median()
            within_mad = price_eval.groupby("cluster")["log_target"].apply(
                lambda series: float(np.median(np.abs(series - np.median(series))))
            )
            within_weights = price_eval.groupby("cluster").size().reindex(within_mad.index).to_numpy()
            weighted_within = float(np.average(within_mad.to_numpy(), weights=within_weights))
            between_std = float(cluster_median.std(ddof=0))
            price_ratio = between_std / max(weighted_within, 1e-6)
            price_score = float(np.tanh(price_ratio / 2.0))

        min_cluster_size = int(counts.min())
        median_cluster_size = float(counts.median())
        max_cluster_size = int(counts.max())
        size_score = float(np.clip(min_cluster_size / max(config.min_cluster_size_main, 1), 0, 1))
        silhouette_norm = float(np.clip((silhouette + 1.0) / 2.0, 0, 1)) if pd.notna(silhouette) else 0.0
        price_norm = float(price_score) if pd.notna(price_score) else 0.0
        diversity_score = float(np.clip(1.0 - dominant_segment_share_mean, 0, 1))
        cluster_score = (
            0.32 * silhouette_norm
            + 0.28 * price_norm
            + 0.18 * mean_confidence
            + 0.12 * size_score
            + 0.10 * diversity_score
        )

        candidate_rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette,
                "price_separation_score": price_score,
                "mean_cluster_confidence": mean_confidence,
                "min_cluster_size": min_cluster_size,
                "median_cluster_size": median_cluster_size,
                "max_cluster_size": max_cluster_size,
                "dominant_segment_share_mean": dominant_segment_share_mean,
                "cluster_score": cluster_score,
                "selected": False,
            }
        )

    report = pd.DataFrame(candidate_rows).sort_values(
        ["cluster_score", "min_cluster_size", "n_clusters"],
        ascending=[False, False, True],
    )
    selected_k = int(report.iloc[0]["n_clusters"])
    report["selected"] = report["n_clusters"] == selected_k
    return selected_k, report.sort_values("n_clusters").reset_index(drop=True)


def similarity_graph_from_features(
    values: np.ndarray,
    n_neighbors: int,
    full_size: int | None = None,
    valid_index: np.ndarray | None = None,
) -> sparse.csr_matrix:
    row_count = values.shape[0]
    if row_count <= 1:
        size = full_size or row_count
        return sparse.csr_matrix((size, size))
    k = min(n_neighbors, row_count - 1)
    graph = kneighbors_graph(values, n_neighbors=k, mode="distance", include_self=False)
    if graph.nnz > 0:
        positive = graph.data[graph.data > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        sigma = max(sigma, 1e-6)
        graph.data = np.exp(-((graph.data ** 2) / (2 * sigma ** 2)))
    graph = ((graph + graph.T) * 0.5).tocsr()
    if full_size is None or valid_index is None:
        return graph
    expanded = sparse.lil_matrix((full_size, full_size))
    row_idx, col_idx = graph.nonzero()
    expanded[valid_index[row_idx], valid_index[col_idx]] = graph[row_idx, col_idx].A1
    return expanded.tocsr()


def build_hybrid_embedding(
    train_df: pd.DataFrame,
    location_feature_cols: list[str],
    feature_bundle: dict[str, Any],
    access_bundle: dict[str, Any],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, KMeans, list[str], pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    core_scaled = transform_scaled_bundle(train, feature_bundle)
    access_scaled = transform_scaled_bundle(train, access_bundle)
    location_scaled = train[location_feature_cols].to_numpy()

    feature_graph = similarity_graph_from_features(core_scaled, n_neighbors=25)
    access_graph = similarity_graph_from_features(access_scaled, n_neighbors=20)
    location_graph = similarity_graph_from_features(location_scaled, n_neighbors=20)

    coord_values = train[["segment_lat_for_model", "segment_lon_for_model"]].to_numpy()
    valid_coord_mask = np.isfinite(coord_values).all(axis=1)
    valid_coord_index = np.where(valid_coord_mask)[0]
    if valid_coord_index.size > 1:
        spatial = coord_values[valid_coord_mask].copy()
        lat_ref = float(np.nanmedian(spatial[:, 0]))
        spatial[:, 0] = spatial[:, 0] * 110_540
        spatial[:, 1] = spatial[:, 1] * 111_320 * math.cos(math.radians(lat_ref))
        spatial_graph = similarity_graph_from_features(
            spatial,
            n_neighbors=25,
            full_size=len(train),
            valid_index=valid_coord_index,
        )
    else:
        spatial_graph = sparse.csr_matrix((len(train), len(train)))

    final_graph = (
        config.alpha["feature"] * feature_graph
        + config.alpha["access"] * access_graph
        + config.alpha["location"] * location_graph
        + config.alpha["spatial"] * spatial_graph
    ).tocsr()
    final_graph = ((final_graph + final_graph.T) * 0.5).tocsr()
    final_graph = final_graph + sparse.identity(len(train), format="csr") * 1e-3

    spectral_dim = min(config.spectral_dim, max(2, len(train) - 2))
    spectral = SpectralEmbedding(
        n_components=spectral_dim,
        affinity="precomputed",
        random_state=config.random_state,
    )
    embedding = spectral.fit_transform(final_graph)
    spectral_cols = [f"spectral_{idx + 1}" for idx in range(embedding.shape[1])]
    for idx, column in enumerate(spectral_cols):
        train[column] = embedding[:, idx]

    cluster_candidate_report_k, cluster_candidate_report = choose_cluster_count(embedding, train, config)
    unique_segments = max(train["segment_key"].nunique(), 3)
    n_clusters = min(8, max(3, unique_segments // 8))
    cluster_candidate_report["selected_production"] = cluster_candidate_report["n_clusters"] == n_clusters
    cluster_candidate_report["selected_by_score"] = cluster_candidate_report["n_clusters"] == cluster_candidate_report_k
    cluster_model = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=40)
    cluster_model.fit(embedding)
    clusters, confidence, assigned_distance = cluster_assignment_details(cluster_model, embedding)
    cluster_distance_scale = (
        pd.DataFrame({"cluster": clusters, "cluster_center_distance_raw": assigned_distance})
        .groupby("cluster")["cluster_center_distance_raw"]
        .quantile(0.75)
        .to_dict()
    )
    normalized_distance = np.array(
        [
            raw_distance / max(float(cluster_distance_scale.get(cluster, 1.0)), 1e-6)
            for cluster, raw_distance in zip(clusters, assigned_distance)
        ]
    )

    train["assigned_cluster"] = clusters.astype(int)
    train["cluster_confidence"] = np.clip(confidence, 0, 1)
    train["cluster_center_distance_raw"] = assigned_distance
    train["distance_to_cluster_center"] = normalized_distance
    train["local_embedding_density"] = graph_local_density(final_graph)
    train["segment_found_in_train"] = True
    train["assigned_cluster_method"] = "SEGMENT_LOOKUP"
    train["case_type"] = "CASE_1_SEGMENT_IN_TRAIN"
    train["cluster"] = train["assigned_cluster"]

    graph_report = pd.DataFrame(
        [
            {
                "train_rows": int(len(train)),
                "n_neighbors_feature": 25,
                "n_neighbors_access": 20,
                "n_neighbors_location": 20,
                "n_neighbors_spatial": 25,
                "alpha_feature": float(config.alpha["feature"]),
                "alpha_access": float(config.alpha["access"]),
                "alpha_location": float(config.alpha["location"]),
                "alpha_spatial": float(config.alpha["spatial"]),
                "spectral_dim": int(len(spectral_cols)),
                "feature_edges": int(feature_graph.nnz // 2),
                "access_edges": int(access_graph.nnz // 2),
                "location_edges": int(location_graph.nnz // 2),
                "spatial_edges": int(spatial_graph.nnz // 2),
                "final_edges": int(final_graph.nnz // 2),
            }
        ]
    )
    return train, cluster_model, spectral_cols, cluster_candidate_report, graph_report


def build_cluster_profile_table(train_df: pd.DataFrame) -> pd.DataFrame:
    global_quantiles = (
        train_df.loc[train_df["tong_gia"] > 0, "tong_gia"].quantile([0.25, 0.5, 0.75]).to_dict()
    )
    profile = (
        train_df.groupby("assigned_cluster")
        .agg(
            cluster_sample_count=("row_id", "size"),
            cluster_distance_scale=("cluster_center_distance_raw", lambda series: float(max(series.quantile(0.75), 1e-6))),
            cluster_mean_confidence=("cluster_confidence", "mean"),
        )
        .reset_index()
        .rename(columns={"assigned_cluster": "cluster"})
    )
    cluster_band = (
        train_df[train_df["tong_gia"] > 0]
        .groupby("assigned_cluster")["tong_gia"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(
            columns={
                0.25: "cluster_total_value_p25",
                0.5: "cluster_total_value_median",
                0.75: "cluster_total_value_p75",
            }
        )
        .reset_index()
        .rename(columns={"assigned_cluster": "cluster"})
    )
    profile = profile.merge(cluster_band, on="cluster", how="left")
    profile["cluster_total_value_p25"] = profile["cluster_total_value_p25"].fillna(float(global_quantiles.get(0.25, np.nan)))
    profile["cluster_total_value_median"] = profile["cluster_total_value_median"].fillna(float(global_quantiles.get(0.5, np.nan)))
    profile["cluster_total_value_p75"] = profile["cluster_total_value_p75"].fillna(float(global_quantiles.get(0.75, np.nan)))
    return profile


def add_cluster_profile_columns(df: pd.DataFrame, cluster_profile_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cluster_key = "assigned_cluster" if "assigned_cluster" in out.columns else "cluster"
    profile = cluster_profile_df.copy()
    out = out.merge(profile, left_on=cluster_key, right_on="cluster", how="left", suffixes=("", "_profile"))
    if "cluster_profile" in out.columns:
        out = out.drop(columns=["cluster_profile"])
    if "cluster_center_distance_raw" in out.columns:
        out["distance_to_cluster_center"] = out["cluster_center_distance_raw"] / np.maximum(
            out["cluster_distance_scale"].fillna(1.0),
            1e-6,
        )
    out["cluster_sample_count"] = out["cluster_sample_count"].fillna(0).astype(int)
    for column in [
        "cluster_total_value_p25",
        "cluster_total_value_median",
        "cluster_total_value_p75",
        "cluster_mean_confidence",
    ]:
        if column not in out.columns:
            out[column] = np.nan
    return out


def add_special_asset_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default)
        return pd.Series(default, index=out.index, dtype=float)

    distance_score = np.clip(col("distance_to_cluster_center") / 1.5, 0, 1)
    low_confidence_score = 1 - np.clip(col("cluster_confidence"), 0, 1)
    sparse_neighbor_score = 1 - np.clip(col("neighbor_count_1000m") / 8.0, 0, 1)
    rarity_score = np.clip(
        0.6 * col("pho_rare") + 0.4 * col("phuong_rare"),
        0,
        1,
    )
    local_density_penalty = 1 - np.clip(col("local_embedding_density"), 0, 1)
    heterogeneity_score = np.clip(col("segment_heterogeneity_score", 0.0), 0, 1.5) / 1.5
    out["special_asset_score"] = np.clip(
        0.28 * low_confidence_score
        + 0.22 * distance_score
        + 0.15 * sparse_neighbor_score
        + 0.12 * rarity_score
        + 0.10 * local_density_penalty
        + 0.08 * heterogeneity_score
        + 0.13 * (1 - np.clip(col("access_score", 0.5), 0, 1)),
        0,
        1,
    )
    out["special_asset_flag"] = np.select(
        [
            out["special_asset_score"] >= 0.70,
            out["special_asset_score"] >= 0.45,
        ],
        [
            "SPECIAL_HIGH",
            "SPECIAL_MEDIUM",
        ],
        default="MAIN_MARKET",
    )
    out["special_asset_reason"] = out.apply(
        lambda row: join_flags(
            [
                "LOW_CLUSTER_CONFIDENCE" if row.get("cluster_confidence", 1.0) < 0.35 else "",
                "CLUSTER_EDGE_POINT" if row.get("distance_to_cluster_center", 0.0) > 1.15 else "",
                "LOW_LOCAL_DENSITY" if row.get("local_embedding_density", 1.0) < 0.18 else "",
                "LOW_SPATIAL_NEIGHBOR_COUNT" if row.get("neighbor_count_1000m", 0) <= 1 else "",
                "RARE_STREET_PATTERN" if row.get("pho_rare", 0) == 1 else "",
                "HETEROGENEOUS_SEGMENT" if str(row.get("segment_heterogeneity_flag", "")) == "HETEROGENEOUS_HIGH" else "",
            ]
        ),
        axis=1,
    )
    return out


def build_segment_embedding_table(
    train_df: pd.DataFrame,
    spectral_cols: list[str],
) -> pd.DataFrame:
    aggregated = train_df.groupby("segment_key").agg(
        segment_lat_for_model=("segment_lat_for_model", "first"),
        segment_lon_for_model=("segment_lon_for_model", "first"),
        segment_lat=("segment_lat", "first"),
        segment_lon=("segment_lon", "first"),
        train_sample_count=("row_id", "size"),
        cluster_confidence=("cluster_confidence", "mean"),
        cluster_center_distance_raw=("cluster_center_distance_raw", "mean"),
        distance_to_cluster_center=("distance_to_cluster_center", "mean"),
        local_embedding_density=("local_embedding_density", "mean"),
        quan=("quan", "first"),
        phuong=("phuong", "first"),
        pho_canonical=("pho_canonical", "first"),
        phuong_norm=("phuong_norm", "first"),
        quan_norm=("quan_norm", "first"),
    ).reset_index()
    spectral_means = train_df.groupby("segment_key")[spectral_cols].mean().reset_index()
    aggregated = aggregated.merge(spectral_means, on="segment_key", how="left")
    segment_cluster = (
        train_df.groupby("segment_key")["assigned_cluster"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
        .reset_index(name="cluster")
    )
    aggregated = aggregated.merge(segment_cluster, on="segment_key", how="left")
    aggregated["segment_found_in_train"] = aggregated["train_sample_count"] > 0
    return aggregated[
        [
            "segment_key",
            *spectral_cols,
            "cluster",
            "cluster_confidence",
            "cluster_center_distance_raw",
            "distance_to_cluster_center",
            "local_embedding_density",
            "segment_lat_for_model",
            "segment_lon_for_model",
            "segment_lat",
            "segment_lon",
            "train_sample_count",
            "segment_found_in_train",
            "quan",
            "phuong",
            "pho_canonical",
            "phuong_norm",
            "quan_norm",
        ]
    ].copy()


def build_segment_neighbors(segment_master: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    valid_segments = segment_master[
        segment_master["segment_lat_for_model"].notna() & segment_master["segment_lon_for_model"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    sigma = config.sigma_spatial_m
    for segment in valid_segments.itertuples(index=False):
        distances: list[tuple[str, float]] = []
        for other in valid_segments.itertuples(index=False):
            if segment.segment_key == other.segment_key:
                continue
            distance_m = haversine_m(
                segment.segment_lat_for_model,
                segment.segment_lon_for_model,
                other.segment_lat_for_model,
                other.segment_lon_for_model,
            )
            if pd.isna(distance_m) or distance_m > max(config.neighbor_radii_m):
                continue
            distances.append((other.segment_key, distance_m))
        distances.sort(key=lambda item: item[1])
        for rank, (neighbor_key, distance_m) in enumerate(distances, start=1):
            bucket = "1500_2000m_fallback"
            if distance_m <= 300:
                bucket = "0_300m"
            elif distance_m <= 500:
                bucket = "300_500m"
            elif distance_m <= 800:
                bucket = "500_800m"
            elif distance_m <= 1000:
                bucket = "800_1000m"
            elif distance_m <= 1500:
                bucket = "1000_1500m"
            rows.append(
                {
                    "segment_key": segment.segment_key,
                    "neighbor_segment_key": neighbor_key,
                    "distance_m": distance_m,
                    "neighbor_rank": rank,
                    "radius_bucket": bucket,
                    "spatial_weight": math.exp(-((distance_m ** 2) / (2 * sigma ** 2))),
                    "neighbor_type": "spatial_segment_neighbor",
                    "neighbor_source": "manual_coordinate_dictionary",
                }
            )
    return pd.DataFrame(rows)


def build_segment_neighbor_features(neighbor_df: pd.DataFrame) -> pd.DataFrame:
    if neighbor_df.empty:
        return pd.DataFrame(
            columns=[
                "segment_key",
                "neighbor_count_300m",
                "neighbor_count_500m",
                "neighbor_count_1000m",
                "nearest_segment_distance_m",
            ]
        )
    return (
        neighbor_df.groupby("segment_key")
        .agg(
            neighbor_count_300m=("distance_m", lambda series: int((series <= 300).sum())),
            neighbor_count_500m=("distance_m", lambda series: int((series <= 500).sum())),
            neighbor_count_1000m=("distance_m", lambda series: int((series <= 1000).sum())),
            nearest_segment_distance_m=("distance_m", "min"),
        )
        .reset_index()
    )


def attach_neighbor_features(df: pd.DataFrame, segment_neighbor_features: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(segment_neighbor_features, on="segment_key", how="left")
    for column in [
        "neighbor_count_300m",
        "neighbor_count_500m",
        "neighbor_count_1000m",
        "nearest_segment_distance_m",
    ]:
        if column not in out.columns:
            out[column] = np.nan
    out["neighbor_count_300m"] = out["neighbor_count_300m"].fillna(0).astype(int)
    out["neighbor_count_500m"] = out["neighbor_count_500m"].fillna(0).astype(int)
    out["neighbor_count_1000m"] = out["neighbor_count_1000m"].fillna(0).astype(int)
    return out


def build_location_fallback_templates(
    segment_embedding_df: pd.DataFrame,
    spectral_cols: list[str],
    context_cols: list[str],
) -> dict[str, Any]:
    fallback = {
        "global": {},
        "phuong": {},
        "quan": {},
    }
    template_cols = spectral_cols + context_cols
    for column in template_cols:
        fallback["global"][column] = float(segment_embedding_df[column].mean())
    fallback["global"]["cluster"] = int(segment_embedding_df["cluster"].mode().iloc[0])
    fallback["global"]["cluster_confidence"] = float(segment_embedding_df["cluster_confidence"].mean())

    for level in ["phuong_norm", "quan_norm"]:
        grouped = (
            segment_embedding_df.groupby(level)[template_cols]
            .mean()
            .reset_index()
        )
        cluster_mode = (
            segment_embedding_df.groupby(level)["cluster"]
            .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
            .reset_index(name="cluster")
        )
        grouped = grouped.merge(cluster_mode, on=level, how="left")
        target_key = "phuong" if level == "phuong_norm" else "quan"
        fallback[target_key] = grouped.set_index(level).to_dict(orient="index")
    return fallback


def assign_inference_representation(
    df: pd.DataFrame,
    segment_embedding_df: pd.DataFrame,
    segment_neighbor_features: pd.DataFrame,
    spectral_cols: list[str],
    fallback_templates: dict[str, Any],
    cluster_model: KMeans,
    config: PipelineConfig,
) -> pd.DataFrame:
    out = df.copy()
    segment_lookup = segment_embedding_df.set_index("segment_key").to_dict(orient="index")
    neighbor_lookup = segment_neighbor_features.set_index("segment_key").to_dict(orient="index")
    context_cols = ["cluster_center_distance_raw", "distance_to_cluster_center", "local_embedding_density"]
    train_segments_with_geo = segment_embedding_df[
        segment_embedding_df["segment_lat_for_model"].notna() & segment_embedding_df["segment_lon_for_model"].notna()
    ].copy()

    assigned_cluster = []
    assigned_method = []
    segment_found = []
    case_types = []
    cluster_conf = []
    extra_warning = []
    assigned_spectral: dict[str, list[float]] = {column: [] for column in spectral_cols}
    assigned_context: dict[str, list[float]] = {column: [] for column in context_cols}
    neighbor_300 = []
    neighbor_500 = []
    neighbor_1000 = []
    nearest_distance = []

    for row in out.itertuples(index=False):
        warnings_for_row: list[str] = []
        segment_info = segment_lookup.get(row.segment_key)
        if segment_info and segment_info["train_sample_count"] > 0:
            segment_found.append(True)
            assigned_method.append("SEGMENT_LOOKUP")
            case_types.append("CASE_1_SEGMENT_IN_TRAIN")
            for column in spectral_cols:
                assigned_spectral[column].append(float(segment_info[column]))
            for column in context_cols:
                assigned_context[column].append(float(segment_info.get(column, 0.0)))
            neighbor_features = neighbor_lookup.get(row.segment_key, {})
            neighbor_300.append(int(neighbor_features.get("neighbor_count_300m", 0)))
            neighbor_500.append(int(neighbor_features.get("neighbor_count_500m", 0)))
            neighbor_1000.append(int(neighbor_features.get("neighbor_count_1000m", 0)))
            nearest_distance.append(float(neighbor_features.get("nearest_segment_distance_m", np.nan)))
            spectral_vector = np.array([segment_info[column] for column in spectral_cols], dtype=float).reshape(1, -1)
            cluster_id, confidence, raw_distance = cluster_assignment_details(cluster_model, spectral_vector)
            assigned_cluster.append(int(cluster_id[0]))
            cluster_conf.append(float(confidence[0]))
            assigned_context["cluster_center_distance_raw"][-1] = float(raw_distance[0])
            extra_warning.append("OK")
            continue

        if row.has_valid_coordinate and not train_segments_with_geo.empty:
            train_geo = train_segments_with_geo.copy()
            train_geo["distance_m"] = train_geo.apply(
                lambda item: haversine_m(
                    row.segment_lat_for_model,
                    row.segment_lon_for_model,
                    item["segment_lat_for_model"],
                    item["segment_lon_for_model"],
                ),
                axis=1,
            )
            available = train_geo.sort_values("distance_m")
            selected = pd.DataFrame()
            used_radius = max(config.neighbor_radii_m)
            for radius in config.neighbor_radii_m:
                subset = available[available["distance_m"] <= radius].copy()
                if len(subset) >= config.min_comparable_count or radius == max(config.neighbor_radii_m):
                    selected = subset
                    used_radius = radius
                    break
            if selected.empty:
                selected = available.head(config.top_k_comparable).copy()
                warnings_for_row.extend(["FAR_FROM_KNOWN_SEGMENTS", "OUT_OF_TRAIN_AREA"])
            if used_radius > 1000:
                warnings_for_row.append("FALLBACK_RADIUS_USED")
            if len(selected) < config.min_comparable_count:
                warnings_for_row.append("LOW_NEIGHBOR_COUNT")

            weights = np.exp(-selected["distance_m"].to_numpy() / config.sigma_spatial_m)
            weight_sum = max(weights.sum(), 1e-6)
            segment_found.append(False)
            assigned_method.append("KNN_WEIGHTED_SPATIAL")
            case_types.append("CASE_2_NEW_SEGMENT_WITH_COORD")
            spectral_vector_values = []
            for column in spectral_cols:
                value = float(np.average(selected[column], weights=weights))
                spectral_vector_values.append(value)
                assigned_spectral[column].append(value)
            for column in context_cols:
                assigned_context[column].append(float(np.average(selected[column], weights=weights)))
            spectral_vector = np.array(spectral_vector_values, dtype=float).reshape(1, -1)
            cluster_id, confidence, raw_distance = cluster_assignment_details(cluster_model, spectral_vector)
            assigned_cluster.append(int(cluster_id[0]))
            cluster_conf.append(float(confidence[0]))
            assigned_context["cluster_center_distance_raw"][-1] = float(raw_distance[0])
            neighbor_300.append(int((selected["distance_m"] <= 300).sum()))
            neighbor_500.append(int((selected["distance_m"] <= 500).sum()))
            neighbor_1000.append(int((selected["distance_m"] <= 1000).sum()))
            nearest_distance.append(float(selected["distance_m"].min()))
            extra_warning.append(join_flags(warnings_for_row))
            continue

        segment_found.append(False)
        assigned_method.append("LOCATION_FALLBACK")
        case_types.append("CASE_3_LOCATION_FALLBACK")
        phuong_template = fallback_templates["phuong"].get(row.phuong_norm)
        quan_template = fallback_templates["quan"].get(row.quan_norm)
        template = phuong_template or quan_template or fallback_templates["global"]
        spectral_vector_values = []
        for column in spectral_cols:
            value = float(template.get(column, 0.0))
            spectral_vector_values.append(value)
            assigned_spectral[column].append(value)
        for column in context_cols:
            assigned_context[column].append(float(template.get(column, 0.0)))
        spectral_vector = np.array(spectral_vector_values, dtype=float).reshape(1, -1)
        cluster_id, confidence, raw_distance = cluster_assignment_details(cluster_model, spectral_vector)
        assigned_cluster.append(int(cluster_id[0]))
        cluster_conf.append(float(confidence[0]) * 0.5)
        assigned_context["cluster_center_distance_raw"][-1] = float(raw_distance[0])
        neighbor_300.append(0)
        neighbor_500.append(0)
        neighbor_1000.append(0)
        nearest_distance.append(np.nan)
        extra_warning.append(join_flags(["MISSING_SEGMENT_GEO", "LOW_CONFIDENCE_PREDICTION"]))

    for column in spectral_cols:
        out[column] = assigned_spectral[column]
    for column in context_cols:
        out[column] = assigned_context[column]
    out["segment_found_in_train"] = segment_found
    out["assigned_cluster_method"] = assigned_method
    out["case_type"] = case_types
    out["assigned_cluster"] = assigned_cluster
    out["cluster"] = out["assigned_cluster"]
    out["cluster_confidence"] = cluster_conf
    out["neighbor_count_300m"] = neighbor_300
    out["neighbor_count_500m"] = neighbor_500
    out["neighbor_count_1000m"] = neighbor_1000
    out["nearest_segment_distance_m"] = nearest_distance
    out["inference_warning_flag"] = extra_warning
    out["warning_flag_base"] = out.apply(
        lambda row: join_flags([row["warning_flag_base"], row["inference_warning_flag"]]),
        axis=1,
    )
    return out


def build_distance_lookup(segment_embedding_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    usable = segment_embedding_df[
        segment_embedding_df["segment_lat_for_model"].notna() & segment_embedding_df["segment_lon_for_model"].notna()
    ]
    for left in usable.itertuples(index=False):
        lookup[(left.segment_key, left.segment_key)] = 0.0
        for right in usable.itertuples(index=False):
            if left.segment_key == right.segment_key:
                continue
            lookup[(left.segment_key, right.segment_key)] = haversine_m(
                left.segment_lat_for_model,
                left.segment_lon_for_model,
                right.segment_lat_for_model,
                right.segment_lon_for_model,
            )
    return lookup


def build_comparable_index(
    clean_train_df: pd.DataFrame,
    segment_embedding_df: pd.DataFrame,
    feature_bundle: dict[str, Any],
    access_bundle: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    candidates = clean_train_df.copy().reset_index(drop=True)
    candidates["_candidate_idx"] = np.arange(len(candidates))
    core_matrix = transform_scaled_bundle(candidates, feature_bundle)
    access_matrix = transform_scaled_bundle(candidates, access_bundle)

    segment_to_indices = candidates.groupby("segment_key")["_candidate_idx"].apply(list).to_dict()
    distance_lookup = build_distance_lookup(segment_embedding_df)
    train_segments = segment_embedding_df[segment_embedding_df["train_sample_count"] > 0].copy()

    radius_lookup: dict[str, dict[int, list[str]]] = {}
    for segment_key in train_segments["segment_key"]:
        radius_lookup[segment_key] = {}
        for radius in [0] + config.neighbor_radii_m:
            neighbors = []
            for other_key in train_segments["segment_key"]:
                distance_m = distance_lookup.get((segment_key, other_key), np.inf)
                if distance_m <= radius:
                    neighbors.append(other_key)
            if segment_key not in neighbors:
                neighbors.append(segment_key)
            radius_lookup[segment_key][radius] = neighbors

    return {
        "candidate_df": candidates,
        "core_matrix": core_matrix,
        "access_matrix": access_matrix,
        "feature_bundle": feature_bundle,
        "access_bundle": access_bundle,
        "segment_to_indices": segment_to_indices,
        "distance_lookup": distance_lookup,
        "radius_lookup": radius_lookup,
        "train_segments": train_segments,
        "spectral_cols": [column for column in clean_train_df.columns if column.startswith("spectral_")],
    }


def select_candidate_indices(
    query_row: pd.Series,
    comparable_index: dict[str, Any],
    config: PipelineConfig,
    exclude_row_id: str | None = None,
) -> tuple[list[int], int, list[str]]:
    candidate_df = comparable_index["candidate_df"]
    segment_to_indices = comparable_index["segment_to_indices"]
    warnings_for_row: list[str] = []

    if bool(query_row.get("segment_found_in_train", False)) and query_row["segment_key"] in comparable_index["radius_lookup"]:
        selected_indices: list[int] = []
        used_radius = max(config.neighbor_radii_m)
        for radius in [0] + config.neighbor_radii_m:
            pool: list[int] = []
            for segment_key in comparable_index["radius_lookup"][query_row["segment_key"]][radius]:
                pool.extend(segment_to_indices.get(segment_key, []))
            if exclude_row_id:
                pool = [idx for idx in pool if candidate_df.iloc[idx]["row_id"] != exclude_row_id]
            selected_indices = pool
            used_radius = radius
            if len(pool) >= config.min_comparable_count or radius == max(config.neighbor_radii_m):
                break
        if used_radius > 1000:
            warnings_for_row.append("FALLBACK_RADIUS_USED")
        return selected_indices, used_radius, warnings_for_row

    if bool(query_row.get("has_valid_coordinate", False)):
        train_segments = comparable_index["train_segments"].copy()
        train_segments["distance_m"] = train_segments.apply(
            lambda item: haversine_m(
                query_row["segment_lat_for_model"],
                query_row["segment_lon_for_model"],
                item["segment_lat_for_model"],
                item["segment_lon_for_model"],
            ),
            axis=1,
        )
        train_segments = train_segments.sort_values("distance_m")
        selected_segments = pd.DataFrame()
        used_radius = max(config.neighbor_radii_m)
        for radius in config.neighbor_radii_m:
            subset = train_segments[train_segments["distance_m"] <= radius].copy()
            if len(subset) >= config.min_comparable_count or radius == max(config.neighbor_radii_m):
                selected_segments = subset
                used_radius = radius
                break
        if selected_segments.empty:
            selected_segments = train_segments.head(config.top_k_comparable).copy()
            warnings_for_row.extend(["FAR_FROM_KNOWN_SEGMENTS", "OUT_OF_TRAIN_AREA"])
        if used_radius > 1000:
            warnings_for_row.append("FALLBACK_RADIUS_USED")
        if len(selected_segments) < config.min_comparable_count:
            warnings_for_row.append("LOW_NEIGHBOR_COUNT")
        selected_indices = []
        for segment_key in selected_segments["segment_key"]:
            selected_indices.extend(segment_to_indices.get(segment_key, []))
        return selected_indices, used_radius, warnings_for_row

    phuong_pool = candidate_df[candidate_df["phuong_norm"] == query_row["phuong_norm"]]["_candidate_idx"].tolist()
    if phuong_pool:
        warnings_for_row.append("LOW_CONFIDENCE_PREDICTION")
        return phuong_pool, max(config.neighbor_radii_m), warnings_for_row
    quan_pool = candidate_df[candidate_df["quan_norm"] == query_row["quan_norm"]]["_candidate_idx"].tolist()
    if quan_pool:
        warnings_for_row.append("LOW_CONFIDENCE_PREDICTION")
        return quan_pool, max(config.neighbor_radii_m), warnings_for_row
    warnings_for_row.extend(["MISSING_SEGMENT_GEO", "LOW_CONFIDENCE_PREDICTION"])
    return candidate_df["_candidate_idx"].tolist(), max(config.neighbor_radii_m), warnings_for_row


def build_comparable_features(
    query_df: pd.DataFrame,
    comparable_index: dict[str, Any],
    config: PipelineConfig,
    exclude_self: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = query_df.copy().reset_index(drop=True)
    candidate_df = comparable_index["candidate_df"]
    query_core = transform_scaled_bundle(queries, comparable_index["feature_bundle"])
    query_access = transform_scaled_bundle(queries, comparable_index["access_bundle"])

    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for query_pos, query_row in queries.iterrows():
        exclude_row_id = query_row["row_id"] if exclude_self else None
        candidate_indices, used_radius, warnings_for_row = select_candidate_indices(
            query_row, comparable_index, config, exclude_row_id=exclude_row_id
        )
        candidates = candidate_df.iloc[candidate_indices].copy()
        if candidates.empty:
            candidates = candidate_df.copy()
            warnings_for_row.append("LOW_COMPARABLE_COUNT")

        same_purpose = candidates[candidates["muc_dich_su_dung_dat"] == query_row["muc_dich_su_dung_dat"]].copy()
        if len(same_purpose) >= max(3, config.min_comparable_count - 1):
            candidates = same_purpose
        else:
            warnings_for_row.append("PURPOSE_MATCH_RELAXED")
        same_type = candidates[candidates["phan_loai_kho"] == query_row["phan_loai_kho"]].copy()
        if len(same_type) >= max(3, config.min_comparable_count - 1):
            candidates = same_type
        else:
            warnings_for_row.append("PROPERTY_TYPE_RELAXED")

        same_access = candidates[candidates["access_band"] == query_row["access_band"]].copy()
        access_keep_threshold = max(3, config.min_comparable_count - 2)
        if len(same_access) >= access_keep_threshold:
            candidates = same_access
        else:
            warnings_for_row.append("ACCESS_MATCH_RELAXED")

        area = float(pd.to_numeric(pd.Series([query_row["dien_tich_m2"]]), errors="coerce").iloc[0])
        if not np.isfinite(area) or area <= 0:
            area = float(pd.to_numeric(candidates["dien_tich_m2"], errors="coerce").median())
        area_20 = candidates[candidates["dien_tich_m2"].between(area * 0.8, area * 1.2, inclusive="both")].copy()
        if len(area_20) >= 3:
            candidates = area_20
        else:
            area_35 = candidates[candidates["dien_tich_m2"].between(area * 0.65, area * 1.35, inclusive="both")].copy()
            if len(area_35) >= 3:
                candidates = area_35
                warnings_for_row.append("AREA_MATCH_RELAXED")
            else:
                area_50 = candidates[candidates["dien_tich_m2"].between(area * 0.5, area * 1.5, inclusive="both")].copy()
                if len(area_50) >= 3:
                    candidates = area_50
                    warnings_for_row.append("AREA_MATCH_STRONGLY_RELAXED")

        if candidates.empty:
            candidates = candidate_df.iloc[candidate_indices].copy()

        candidate_positions = candidates["_candidate_idx"].tolist()
        feature_dist = np.linalg.norm(comparable_index["core_matrix"][candidate_positions] - query_core[query_pos], axis=1)
        access_dist = np.linalg.norm(comparable_index["access_matrix"][candidate_positions] - query_access[query_pos], axis=1)
        candidates["feature_similarity"] = 1 / (1 + feature_dist)
        candidates["access_similarity"] = 1 / (1 + access_dist)
        candidates["area_similarity"] = np.exp(
            -np.abs(np.log((candidates["dien_tich_m2"] + 1e-6) / max(area, 1e-6)))
        )
        candidates["same_cluster"] = (candidates["cluster"] == query_row["cluster"]).astype(int)
        candidates["same_access_band"] = (candidates["access_band"] == query_row["access_band"]).astype(int)
        candidates["same_segment"] = (candidates["segment_key"] == query_row["segment_key"]).astype(int)
        candidates["same_phuong"] = (candidates["phuong_norm"] == query_row["phuong_norm"]).astype(int)
        if "special_asset_score" not in candidates.columns:
            candidates["special_asset_score"] = 0.0
        if "segment_heterogeneity_penalty" not in candidates.columns:
            candidates["segment_heterogeneity_penalty"] = 0.85
        if "segment_heterogeneity_flag" not in candidates.columns:
            candidates["segment_heterogeneity_flag"] = "UNKNOWN_SEGMENT"
        candidates["special_gap"] = np.abs(
            candidates.get("special_asset_score", 0.0).fillna(0.0) - float(query_row.get("special_asset_score", 0.0))
        )
        candidates["location_similarity"] = candidates.apply(
            lambda row: 1.0
            if row["segment_key"] == query_row["segment_key"] and query_row["segment_key"]
            else 0.9
            if row["pho_canonical_norm"] == query_row["pho_canonical_norm"] and query_row["pho_canonical_norm"]
            else 0.6
            if row["phuong_norm"] == query_row["phuong_norm"]
            else 0.4
            if row["quan_norm"] == query_row["quan_norm"]
            else 0.2,
            axis=1,
        )

        if bool(query_row.get("has_valid_coordinate", False)):
            candidates["distance_m"] = candidates.apply(
                lambda row: haversine_m(
                    query_row["segment_lat_for_model"],
                    query_row["segment_lon_for_model"],
                    row["segment_lat_for_model"],
                    row["segment_lon_for_model"],
                ),
                axis=1,
            )
        else:
            candidates["distance_m"] = candidates["segment_key"].map(
                lambda key: comparable_index["distance_lookup"].get((query_row["segment_key"], key), np.inf)
            )
        candidates["spatial_similarity"] = np.where(
            np.isfinite(candidates["distance_m"]),
            np.exp(-candidates["distance_m"] / config.sigma_spatial_m),
            0.0,
        )

        query_frontage = pd.to_numeric(pd.Series([query_row.get("mat_tien_m")]), errors="coerce").iloc[0]
        query_depth = pd.to_numeric(pd.Series([query_row.get("chieu_dai_m")]), errors="coerce").iloc[0]
        query_lt = pd.to_numeric(pd.Series([query_row.get("lt_chuan")]), errors="coerce").iloc[0]
        query_shape = pd.to_numeric(pd.Series([query_row.get("chi_so_hinhdang")]), errors="coerce").iloc[0]
        if pd.isna(query_frontage) or query_frontage <= 0:
            query_frontage = float(pd.to_numeric(candidates["mat_tien_m"], errors="coerce").median())
        if pd.isna(query_depth) or query_depth <= 0:
            query_depth = float(pd.to_numeric(candidates["chieu_dai_m"], errors="coerce").median())
        if pd.isna(query_lt):
            query_lt = float(pd.to_numeric(candidates["lt_chuan"], errors="coerce").median())
        if pd.isna(query_shape):
            query_shape = float(pd.to_numeric(candidates["chi_so_hinhdang"], errors="coerce").median())
        if pd.isna(query_frontage) or query_frontage <= 0:
            query_frontage = 1.0
        if pd.isna(query_depth) or query_depth <= 0:
            query_depth = 1.0
        if pd.isna(query_lt):
            query_lt = 0.0
        if pd.isna(query_shape):
            query_shape = 0.0

        candidates["frontage_similarity"] = np.exp(
            -np.abs(
                np.log(
                    (pd.to_numeric(candidates["mat_tien_m"], errors="coerce").fillna(0) + 1e-6)
                    / max(float(query_frontage) + 1e-6, 1e-6)
                )
            )
        )
        candidates["depth_similarity"] = np.exp(
            -np.abs(
                np.log(
                    (pd.to_numeric(candidates["chieu_dai_m"], errors="coerce").fillna(0) + 1e-6)
                    / max(float(query_depth) + 1e-6, 1e-6)
                )
            )
        )
        candidates["lt_similarity"] = np.exp(
            -np.abs(
                pd.to_numeric(candidates["lt_chuan"], errors="coerce").fillna(0)
                - float(query_lt)
            )
        )
        candidates["shape_similarity"] = np.exp(
            -np.abs(
                pd.to_numeric(candidates["chi_so_hinhdang"], errors="coerce").fillna(0)
                - float(query_shape)
            )
        )

        candidates["comparable_source"] = candidates.apply(
            lambda row: "same_segment"
            if row["segment_key"] == query_row["segment_key"] and query_row["segment_key"]
            else "nearby_segment"
            if pd.notna(row["distance_m"]) and row["distance_m"] <= 1000
            else "same_cluster"
            if row["cluster"] == query_row["cluster"]
            else "same_phuong"
            if row["phuong_norm"] == query_row["phuong_norm"]
            else "fallback",
            axis=1,
        )
        source_score_map = {
            "same_segment": 1.00,
            "nearby_segment": 0.90,
            "same_cluster": 0.75,
            "same_phuong": 0.68,
            "fallback": 0.55,
        }
        candidates["comparable_source_score"] = candidates["comparable_source"].map(source_score_map).fillna(0.55)

        sample_count_score = np.clip(pd.to_numeric(candidates["pho_count_train"], errors="coerce").fillna(0) / 10.0, 0.3, 1.0)
        access_band_score = np.where(candidates["same_access_band"] == 1, 1.0, 0.7)
        cluster_conf_score = np.clip(pd.to_numeric(candidates["cluster_confidence"], errors="coerce").fillna(0.3), 0.2, 1.0)
        candidates["candidate_reliability"] = np.clip(
            (
                0.30 * sample_count_score
                + 0.25 * candidates["comparable_source_score"]
                + 0.25 * cluster_conf_score
                + 0.20 * access_band_score
            )
            * pd.to_numeric(candidates["segment_heterogeneity_penalty"], errors="coerce").fillna(0.85)
            * (1 - 0.08 * candidates["special_asset_score"].clip(lower=0, upper=1)),
            0,
            1,
        )

        distance_penalty = np.where(
            np.isfinite(candidates["distance_m"]),
            np.clip(candidates["distance_m"] / 2000, 0, 1),
            1.0,
        )
        area_penalty = np.clip(np.abs(np.log((candidates["dien_tich_m2"] + 1e-6) / max(area, 1e-6))), 0, 1)
        special_penalty = candidates["special_gap"].clip(lower=0, upper=1)
        access_mismatch_penalty = 1 - candidates["same_access_band"]

        same_segment_mask = candidates["comparable_source"] == "same_segment"
        nearby_mask = candidates["comparable_source"] == "nearby_segment"
        same_cluster_mask = candidates["comparable_source"] == "same_cluster"

        same_segment_score = (
            0.24 * candidates["access_similarity"]
            + 0.20 * candidates["feature_similarity"]
            + 0.16 * candidates["frontage_similarity"]
            + 0.12 * candidates["area_similarity"]
            + 0.10 * candidates["depth_similarity"]
            + 0.10 * candidates["candidate_reliability"]
            + 0.08 * candidates["location_similarity"]
        )
        nearby_segment_score = (
            0.24 * candidates["spatial_similarity"]
            + 0.18 * candidates["access_similarity"]
            + 0.16 * candidates["feature_similarity"]
            + 0.12 * candidates["location_similarity"]
            + 0.10 * candidates["same_cluster"]
            + 0.10 * candidates["candidate_reliability"]
            + 0.10 * candidates["area_similarity"]
        )
        fallback_score = (
            0.18 * candidates["spatial_similarity"]
            + 0.16 * candidates["location_similarity"]
            + 0.16 * candidates["access_similarity"]
            + 0.15 * candidates["feature_similarity"]
            + 0.10 * candidates["same_cluster"]
            + 0.10 * candidates["candidate_reliability"]
            + 0.08 * candidates["area_similarity"]
            + 0.07 * candidates["frontage_similarity"]
        )
        candidates["similarity_score"] = np.where(
            same_segment_mask,
            same_segment_score,
            np.where(nearby_mask, nearby_segment_score, fallback_score),
        )
        candidates["similarity_score"] = (
            candidates["similarity_score"]
            - 0.06 * distance_penalty
            - 0.05 * area_penalty
            - 0.04 * special_penalty
            - 0.04 * access_mismatch_penalty
            - 0.04 * np.where(
                (str(query_row.get("segment_heterogeneity_flag", "")) == "HETEROGENEOUS_HIGH")
                & (candidates["same_access_band"] == 0),
                1.0,
                0.0,
            )
        )
        candidates["band_weight"] = np.clip(
            candidates["similarity_score"].clip(lower=0) * (0.55 + 0.45 * candidates["candidate_reliability"]),
            0,
            None,
        )

        topk = candidates.sort_values(["similarity_score", "distance_m"], ascending=[False, True]).head(config.top_k_comparable)
        band_topk = candidates.sort_values(["band_weight", "distance_m"], ascending=[False, True]).head(
            max(config.top_k_comparable, config.band_top_k)
        )
        if len(topk) < config.min_comparable_count:
            warnings_for_row.append("LOW_COMPARABLE_COUNT")

        same_access_band_ratio = float(band_topk["same_access_band"].mean()) if not band_topk.empty else 0.0
        if same_access_band_ratio < 0.5:
            warnings_for_row.append("ACCESS_MISMATCH_POOL")

        for comparable_row in topk.itertuples(index=False):
            result_rows.append(
                {
                    "input_id": query_row["row_id"],
                    "input_dataset": query_row["dataset_name"],
                    "comparable_row_id": comparable_row.row_id,
                    "comparable_segment_key": comparable_row.segment_key,
                    "comparable_source": comparable_row.comparable_source,
                    "distance_m": comparable_row.distance_m,
                    "feature_similarity": comparable_row.feature_similarity,
                    "access_similarity": comparable_row.access_similarity,
                    "area_similarity": comparable_row.area_similarity,
                    "same_cluster": int(comparable_row.same_cluster),
                    "same_access_band": int(comparable_row.same_access_band),
                    "candidate_reliability": comparable_row.candidate_reliability,
                    "similarity_score": comparable_row.similarity_score,
                    "comparable_total_value": comparable_row.tong_gia,
                }
            )

        if band_topk.empty:
            p25 = np.nan
            median = np.nan
            p75 = np.nan
            comparable_confidence = 0.0
            same_cluster_ratio = 0.0
            access_band_match_ratio = 0.0
            special_comparable_ratio = 0.0
            band_confidence = 0.0
            mean_candidate_reliability = 0.0
            same_segment_ratio = 0.0
            heterogeneous_comparable_ratio = 0.0
            weighted_band_width_ratio = np.nan
        else:
            weights = band_topk["band_weight"].to_numpy()
            p25 = weighted_quantile(band_topk["tong_gia"], 0.25, weights)
            median = weighted_quantile(band_topk["tong_gia"], 0.50, weights)
            p75 = weighted_quantile(band_topk["tong_gia"], 0.75, weights)
            same_cluster_ratio = float(band_topk["same_cluster"].mean())
            access_band_match_ratio = float(band_topk["same_access_band"].mean())
            special_comparable_ratio = float((band_topk["special_asset_score"] >= 0.45).mean())
            mean_candidate_reliability = float(band_topk["candidate_reliability"].mean())
            same_segment_ratio = float(band_topk["same_segment"].mean())
            heterogeneous_comparable_ratio = float(
                band_topk["segment_heterogeneity_flag"].isin(["HETEROGENEOUS_MEDIUM", "HETEROGENEOUS_HIGH"]).mean()
            )
            score_strength = float(band_topk["similarity_score"].clip(lower=0).mean())
            count_score = min(1.0, len(band_topk) / max(config.band_top_k, 1))
            distance_score = 1.0 - min(1.0, used_radius / max(config.neighbor_radii_m))
            comparable_confidence = float(
                np.clip(0.45 * count_score + 0.35 * score_strength + 0.20 * distance_score, 0, 1)
            )
            weighted_band_width_ratio = (
                (p75 - p25) / max(median, 1.0)
                if pd.notna(p25) and pd.notna(p75) and pd.notna(median)
                else np.nan
            )
            band_confidence = float(
                np.clip(
                    0.35 * count_score
                    + 0.25 * mean_candidate_reliability
                    + 0.20 * access_band_match_ratio
                    + 0.10 * score_strength
                    + 0.10 * (1 - min(1.0, weighted_band_width_ratio if pd.notna(weighted_band_width_ratio) else 1.0)),
                    0,
                    1,
                )
            )
            if access_band_match_ratio < 0.4:
                warnings_for_row.append("ACCESS_CONTEXT_WEAK")
            if heterogeneous_comparable_ratio > 0.6:
                warnings_for_row.append("HETEROGENEOUS_COMPARABLE_POOL")
            if pd.notna(weighted_band_width_ratio) and weighted_band_width_ratio > 1.2:
                warnings_for_row.append("WIDE_PRICE_BAND")

        summary_rows.append(
            {
                "row_id": query_row["row_id"],
                "total_value_band_p25": p25,
                "total_value_band_median": median,
                "total_value_band_p75": p75,
                "comparable_count": int(len(topk)),
                "comparable_confidence": comparable_confidence,
                "same_cluster_ratio": same_cluster_ratio,
                "access_band_match_ratio": access_band_match_ratio,
                "special_comparable_ratio": special_comparable_ratio,
                "band_confidence": band_confidence,
                "mean_candidate_reliability": mean_candidate_reliability,
                "same_segment_ratio": same_segment_ratio,
                "heterogeneous_comparable_ratio": heterogeneous_comparable_ratio,
                "weighted_band_width_ratio": weighted_band_width_ratio,
                "comparable_warning_flag": join_flags(warnings_for_row),
                "max_search_radius_m": used_radius,
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(summary_rows)


def build_segment_total_value_band(clean_train_df: pd.DataFrame) -> pd.DataFrame:
    cluster_band = (
        clean_train_df.groupby("cluster")["tong_gia"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(
            columns={
                0.25: "cluster_total_value_p25",
                0.5: "cluster_total_value_median",
                0.75: "cluster_total_value_p75",
            }
        )
        .reset_index()
    )
    segment_band = (
        clean_train_df.groupby("segment_key")
        .agg(
            sample_count=("row_id", "size"),
            total_value_p25=("tong_gia", lambda series: float(series.quantile(0.25))),
            total_value_median=("tong_gia", "median"),
            total_value_p75=("tong_gia", lambda series: float(series.quantile(0.75))),
            cluster_main=("cluster", lambda series: int(pd.Series(series).mode().iloc[0])),
        )
        .reset_index()
    )
    segment_band = segment_band.merge(cluster_band, left_on="cluster_main", right_on="cluster", how="left")
    segment_band = segment_band.drop(columns=["cluster"])
    dispersion = (
        (segment_band["total_value_p75"] - segment_band["total_value_p25"])
        / np.maximum(segment_band["total_value_median"], 1)
    )
    count_score = np.clip(segment_band["sample_count"] / 10, 0, 1)
    dispersion_score = np.clip(1 - dispersion, 0, 1)
    segment_band["confidence_score"] = np.clip(0.6 * count_score + 0.4 * dispersion_score, 0, 1)
    return segment_band


def add_segment_band_columns(df: pd.DataFrame, segment_band_df: pd.DataFrame) -> pd.DataFrame:
    band = segment_band_df[
        [
            "segment_key",
            "total_value_p25",
            "total_value_median",
            "total_value_p75",
            "confidence_score",
        ]
    ].rename(
        columns={
            "total_value_p25": "segment_total_value_p25",
            "total_value_median": "segment_total_value_median",
            "total_value_p75": "segment_total_value_p75",
            "confidence_score": "segment_band_confidence_score",
        }
    )
    return df.merge(band, on="segment_key", how="left")


def model_feature_layout(
    location_feature_cols: list[str],
    spectral_cols: list[str],
) -> dict[str, Any]:
    numeric_base = CORE_FEATURE_COLS + ACCESS_FEATURE_COLS + ["nam", "pho_rare", "phuong_rare"]
    numeric_position = POSITION_FEATURE_COLS + ["assigned_cluster", "cluster_confidence"]
    categorical = [
        "phan_loai_kho",
        "muc_dich_su_dung_dat",
        "access_band",
        "segment_heterogeneity_flag",
        "special_asset_flag",
    ]
    return {
        "categorical_features": categorical,
        "sets": {
            "model_1_core_access": numeric_base,
            "model_2_location": numeric_base + location_feature_cols,
            "model_3_location_position": numeric_base + location_feature_cols + POSITION_FEATURE_COLS,
            "model_4_hybrid_embedding": numeric_base
            + location_feature_cols
            + numeric_position
            + spectral_cols,
            "model_5_comparable": numeric_base
            + location_feature_cols
            + numeric_position
            + spectral_cols
            + COMPARABLE_FEATURE_COLS,
        },
    }


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )


def metrics_dict(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    return {
        "mae_total_value": float(mean_absolute_error(actual, pred)),
        "rmse_total_value": float(math.sqrt(mean_squared_error(actual, pred))),
        "mape_total_value": safe_mape(actual, pred),
        "median_absolute_error_total_value": float(median_absolute_error(actual, pred)),
        "mean_log_error": mean_abs_log_error(actual, pred),
    }


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def fit_model_and_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> tuple[HistGradientBoostingRegressor, ColumnTransformer, np.ndarray]:
    clean_train = train_df[train_df["tong_gia"] > 0].copy()
    y_train = np.log1p(clean_train["tong_gia"])
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(clean_train[numeric_features + categorical_features])
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=12,
        l2_regularization=0.05,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    X_pred = preprocessor.transform(predict_df[numeric_features + categorical_features])
    pred = np.expm1(model.predict(X_pred))
    return model, preprocessor, pred


def build_oof_predictions(
    train_df: pd.DataFrame,
    segment_embedding_df: pd.DataFrame,
    feature_bundle: dict[str, Any],
    access_bundle: dict[str, Any],
    feature_layout: dict[str, Any],
    config: PipelineConfig,
) -> pd.DataFrame:
    clean_train = train_df[train_df["tong_gia"] > 0].copy().reset_index(drop=True)
    folds = KFold(n_splits=config.kfold_splits, shuffle=True, random_state=config.random_state)
    records = []
    comparable_cols = COMPARABLE_FEATURE_COLS + ["comparable_warning_flag", "max_search_radius_m"]

    for fold_id, (fit_idx, val_idx) in enumerate(folds.split(clean_train)):
        fold_train = clean_train.iloc[fit_idx].copy().reset_index(drop=True)
        fold_val = clean_train.iloc[val_idx].copy().reset_index(drop=True)
        fold_train = fold_train.drop(columns=[col for col in comparable_cols if col in fold_train.columns], errors="ignore")
        fold_val = fold_val.drop(columns=[col for col in comparable_cols if col in fold_val.columns], errors="ignore")
        fold_index = build_comparable_index(fold_train, segment_embedding_df, feature_bundle, access_bundle, config)
        _, fold_train_summary = build_comparable_features(fold_train, fold_index, config, exclude_self=True)
        _, fold_val_summary = build_comparable_features(fold_val, fold_index, config, exclude_self=False)
        fold_train = fold_train.merge(fold_train_summary, on="row_id", how="left")
        fold_val = fold_val.merge(fold_val_summary, on="row_id", how="left")

        numeric_features = feature_layout["sets"]["model_5_comparable"]
        categorical_features = feature_layout["categorical_features"]
        _, preprocessor, _ = fit_model_and_predict(
            fold_train,
            fold_train,
            numeric_features,
            categorical_features,
            config.random_state + fold_id,
        )
        y_fit = np.log1p(fold_train["tong_gia"])
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=12,
            l2_regularization=0.05,
            random_state=config.random_state + fold_id,
        )
        X_fit = preprocessor.transform(fold_train[numeric_features + categorical_features])
        model.fit(X_fit, y_fit)
        X_val = preprocessor.transform(fold_val[numeric_features + categorical_features])
        fold_pred = np.expm1(model.predict(X_val))
        fold_output = fold_val[
            [
                "row_id",
                "segment_key",
                "pho_canonical",
                "nam",
                "access_band",
                "case_type",
                "assigned_cluster",
                "cluster",
                "cluster_confidence",
                "neighbor_count_1000m",
                "coord_warning_flag",
                "location_warning_flag",
                "warning_flag_base",
                "special_asset_flag",
                "special_asset_score",
                "segment_heterogeneity_flag",
                "total_value_band_p25",
                "total_value_band_median",
                "total_value_band_p75",
                "comparable_count",
                "comparable_confidence",
                "same_cluster_ratio",
                "access_band_match_ratio",
                "band_confidence",
                "mean_candidate_reliability",
                "weighted_band_width_ratio",
                "comparable_warning_flag",
                "max_search_radius_m",
            ]
        ].copy()
        fold_output["actual_total_value"] = fold_val["tong_gia"]
        fold_output["pred_total_value"] = fold_pred
        fold_output["fold"] = fold_id
        fold_output["prediction_source"] = "oof_prediction"
        records.append(fold_output)
    return pd.concat(records, ignore_index=True)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["actual_total_value"] = out.get("actual_total_value", out.get("tong_gia"))
    out["abs_error"] = np.where(
        out["actual_total_value"].notna(),
        np.abs(out["actual_total_value"] - out["pred_total_value"]),
        np.nan,
    )
    out["ape"] = np.where(
        out["actual_total_value"].notna() & (out["actual_total_value"] > 0),
        out["abs_error"] / out["actual_total_value"],
        np.nan,
    )
    out["log_error"] = np.where(
        out["actual_total_value"].notna() & (out["actual_total_value"] > 0) & (out["pred_total_value"] > 0),
        np.abs(np.log1p(out["actual_total_value"]) - np.log1p(out["pred_total_value"])),
        np.nan,
    )
    return out


def add_confidence_and_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def confidence_score(row: pd.Series) -> float:
        geocode_score = 1.0 if row.get("coord_warning_flag", "OK") == "OK" else 0.25
        location_score = 0.2 if row.get("location_warning_flag", "OK") == "MISSING_STREET_NAME" else 1.0
        if "STREET_ALIAS_MERGED" in str(row.get("location_warning_flag", "")):
            location_score = min(location_score, 0.85)
        spatial_score = min(1.0, row.get("neighbor_count_1000m", 0) / 8)
        comparable_count_score = min(1.0, row.get("comparable_count", 0) / 5)
        area_score = 0.7 if "AREA_MATCH_RELAXED" in str(row.get("comparable_warning_flag", "")) else 1.0
        road_evidence_score = float(row.get("road_evidence_confidence", 0.0))
        access_context_score = float(row.get("access_band_match_ratio", 0.0))
        special_penalty_signal = float(row.get("special_asset_score", 0.0))
        band_confidence = float(row.get("band_confidence", row.get("comparable_confidence", 0.0)))
        reliability_score = float(row.get("mean_candidate_reliability", row.get("comparable_confidence", 0.0)))
        heterogeneity_penalty = 0.0
        if row.get("segment_heterogeneity_flag") == "HETEROGENEOUS_HIGH":
            heterogeneity_penalty = 0.10
        elif row.get("segment_heterogeneity_flag") == "HETEROGENEOUS_MEDIUM":
            heterogeneity_penalty = 0.05
        elif row.get("segment_heterogeneity_flag") == "LOW_SAMPLE_SEGMENT":
            heterogeneity_penalty = 0.06
        dispersion = (
            (row.get("total_value_band_p75", np.nan) - row.get("total_value_band_p25", np.nan))
            / max(row.get("total_value_band_median", np.nan), 1)
            if pd.notna(row.get("total_value_band_median", np.nan))
            else 1.5
        )
        price_band_score = float(np.clip(1 - dispersion, 0, 1))
        cluster_score = float(row.get("same_cluster_ratio", 0.0))
        case_penalty = 0.0
        if row.get("case_type") == "CASE_2_NEW_SEGMENT_WITH_COORD":
            case_penalty = 0.08
        elif row.get("case_type") == "CASE_3_LOCATION_FALLBACK":
            case_penalty = 0.18
        score = (
            0.18 * geocode_score
            + 0.13 * location_score
            + 0.12 * spatial_score
            + 0.18 * comparable_count_score
            + 0.09 * area_score
            + 0.10 * price_band_score
            + 0.10 * cluster_score
            + 0.08 * access_context_score
            + 0.08 * band_confidence
            + 0.07 * reliability_score
            + 0.10 * road_evidence_score
        )
        warning_text = join_flags(
            [row.get("warning_flag_base", "OK"), row.get("comparable_warning_flag", "OK")]
        )
        penalty = case_penalty
        if "WARD_COORD_OUTLIER" in warning_text or "MISSING_SEGMENT_GEO" in warning_text:
            penalty += 0.15
        if "LOW_COMPARABLE_COUNT" in warning_text:
            penalty += 0.10
        if "LOW_CONFIDENCE_PREDICTION" in warning_text:
            penalty += 0.10
        if "ACCESS_MISMATCH_POOL" in warning_text or "ACCESS_CONTEXT_WEAK" in warning_text:
            penalty += 0.08
        if "HETEROGENEOUS_COMPARABLE_POOL" in warning_text:
            penalty += 0.06
        if "WIDE_PRICE_BAND" in warning_text:
            penalty += 0.06
        penalty += heterogeneity_penalty
        penalty += 0.10 * special_penalty_signal
        return float(np.clip(score - penalty, 0, 1))

    out["confidence_score"] = out.apply(confidence_score, axis=1)
    out["warning_flag"] = out.apply(
        lambda row: join_flags([row.get("warning_flag_base", "OK"), row.get("comparable_warning_flag", "OK")]),
        axis=1,
    )
    out["explanation_text"] = out.apply(
        lambda row: (
            f"{row.get('pho_canonical', '') or 'Missing street'} | "
            f"{row.get('case_type', 'UNKNOWN_CASE')} | "
            f"cluster {int(row.get('assigned_cluster', row.get('cluster', -1)))} by {row.get('assigned_cluster_method', 'NA')} | "
            f"access {row.get('access_band', 'unknown')} | "
            f"band P25/Med/P75 {format_money(row.get('total_value_band_p25', np.nan))}/"
            f"{format_money(row.get('total_value_band_median', np.nan))}/"
            f"{format_money(row.get('total_value_band_p75', np.nan))} | "
            f"{int(row.get('comparable_count', 0))} comparable(s) within {int(row.get('max_search_radius_m', 0))}m | "
            f"band_conf {row.get('band_confidence', row.get('comparable_confidence', 0)):.2f} | "
            f"hetero {row.get('segment_heterogeneity_flag', 'NA')} | "
            f"special {row.get('special_asset_flag', 'NA')} ({row.get('special_asset_score', 0):.2f}) | "
            f"confidence {row.get('confidence_score', 0):.2f} | "
            f"warnings {row.get('warning_flag', 'OK')}"
        ),
        axis=1,
    )
    return add_error_columns(out)


def confidence_group(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium_high"
    if score >= 0.4:
        return "medium"
    return "low"


def build_group_metrics(df: pd.DataFrame, group_col: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = {
            group_col: group_value,
            "row_count": int(len(group_df)),
            **metrics_dict(group_df["actual_total_value"], group_df["pred_total_value"]),
        }
        if extra_cols:
            for column in extra_cols:
                row[column] = float(group_df[column].mean()) if column in group_df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_diagnostic_reports(pred_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir / "output_1")
    eval_df = pred_df[pred_df["actual_total_value"].notna()].copy()
    if eval_df.empty:
        return
    eval_df["confidence_group"] = eval_df["confidence_score"].map(confidence_group)
    eval_df["confidence_bucket"] = pd.cut(
        eval_df["confidence_score"],
        bins=[-0.001, 0.4, 0.7, 1.0],
        labels=["0.0_0.4", "0.4_0.7", "0.7_1.0"],
    )
    build_group_metrics(eval_df, "nam").rename(columns={"nam": "nam"}).to_csv(
        out_dir / "error_by_year.csv", index=False
    )
    build_group_metrics(eval_df, "segment_key", extra_cols=["confidence_score"]).merge(
        eval_df.groupby("segment_key")[["pho_canonical"]].first().reset_index(),
        on="segment_key",
        how="left",
    ).to_csv(out_dir / "error_by_segment.csv", index=False)
    build_group_metrics(eval_df, "assigned_cluster", extra_cols=["confidence_score"]).rename(
        columns={"assigned_cluster": "cluster"}
    ).to_csv(out_dir / "error_by_cluster.csv", index=False)
    if "access_band" in eval_df.columns:
        build_group_metrics(eval_df, "access_band", extra_cols=["confidence_score"]).to_csv(
            out_dir / "error_by_access_band.csv",
            index=False,
        )
    if "special_asset_flag" in eval_df.columns:
        build_group_metrics(eval_df, "special_asset_flag", extra_cols=["confidence_score"]).to_csv(
            out_dir / "error_by_special_asset_flag.csv",
            index=False,
        )
    if "segment_heterogeneity_flag" in eval_df.columns:
        build_group_metrics(eval_df, "segment_heterogeneity_flag", extra_cols=["confidence_score"]).to_csv(
            out_dir / "error_by_heterogeneous_segment.csv",
            index=False,
        )
    build_group_metrics(eval_df, "confidence_group").to_csv(out_dir / "error_by_confidence_group.csv", index=False)
    build_group_metrics(eval_df, "case_type").to_csv(out_dir / "error_by_case.csv", index=False)
    confidence_calibration_report = (
        eval_df.groupby("confidence_bucket", observed=False, dropna=False)
        .agg(
            row_count=("row_id", "size"),
            mean_confidence=("confidence_score", "mean"),
            mape_total_value=("ape", "mean"),
            median_ape=("ape", "median"),
            band_coverage=(
                "row_id",
                lambda series: float(
                    (
                        eval_df.loc[series.index, "actual_total_value"].between(
                            eval_df.loc[series.index, "total_value_band_p25"],
                            eval_df.loc[series.index, "total_value_band_p75"],
                            inclusive="both",
                        )
                    ).mean()
                ),
            ),
        )
        .reset_index()
    )
    confidence_calibration_report.to_csv(out_dir / "confidence_calibration_report.csv", index=False)
    comparable_quality_cols = [
        "row_id",
        "segment_key",
        "pho_canonical",
        "comparable_count",
        "comparable_confidence",
        "band_confidence",
        "mean_candidate_reliability",
        "same_cluster_ratio",
        "access_band_match_ratio",
        "same_segment_ratio",
        "heterogeneous_comparable_ratio",
        "weighted_band_width_ratio",
        "warning_flag",
    ]
    comparable_quality_cols = [col for col in comparable_quality_cols if col in eval_df.columns]
    eval_df[comparable_quality_cols].to_csv(out_dir / "comparable_quality_report.csv", index=False)
    pd.DataFrame([metrics_dict(eval_df["actual_total_value"], eval_df["pred_total_value"])]).to_csv(
        out_dir / "test_metrics.csv", index=False
    )


def run_ablation(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_layout: dict[str, Any],
    config: PipelineConfig,
) -> pd.DataFrame:
    if eval_df.empty or eval_df["actual_total_value"].isna().all():
        return pd.DataFrame(
            columns=[
                "model_name",
                "feature_set",
                "train_rows",
                "test_rows",
                "mae_total_value",
                "rmse_total_value",
                "mape_total_value",
                "median_absolute_error_total_value",
                "mean_log_error",
                "case1_mae",
                "case2_mae",
                "high_confidence_mape",
                "low_confidence_mape",
            ]
        )

    rows = []
    categorical_features = feature_layout["categorical_features"]
    clean_eval = eval_df[eval_df["actual_total_value"].notna()].copy()
    for model_name, numeric_features in feature_layout["sets"].items():
        _, _, pred = fit_model_and_predict(
            train_df,
            clean_eval,
            numeric_features,
            categorical_features,
            config.random_state,
        )
        metrics = metrics_dict(clean_eval["actual_total_value"], pd.Series(pred))
        case1_mask = clean_eval["case_type"] == "CASE_1_SEGMENT_IN_TRAIN"
        case2_mask = clean_eval["case_type"] == "CASE_2_NEW_SEGMENT_WITH_COORD"
        high_mask = clean_eval["confidence_score"] >= 0.8
        low_mask = clean_eval["confidence_score"] < 0.4
        rows.append(
            {
                "model_name": model_name,
                "feature_set": "|".join(numeric_features),
                "train_rows": int((train_df["tong_gia"] > 0).sum()),
                "test_rows": int(len(clean_eval)),
                **metrics,
                "case1_mae": float(mean_absolute_error(clean_eval.loc[case1_mask, "actual_total_value"], pred[case1_mask]))
                if case1_mask.any()
                else np.nan,
                "case2_mae": float(mean_absolute_error(clean_eval.loc[case2_mask, "actual_total_value"], pred[case2_mask]))
                if case2_mask.any()
                else np.nan,
                "high_confidence_mape": safe_mape(clean_eval.loc[high_mask, "actual_total_value"], pd.Series(pred[high_mask]))
                if high_mask.any()
                else np.nan,
                "low_confidence_mape": safe_mape(clean_eval.loc[low_mask, "actual_total_value"], pd.Series(pred[low_mask]))
                if low_mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_artifacts(
    config: PipelineConfig,
    canonical_lookup: dict[str, str],
    feature_layout: dict[str, Any],
    frequency_lookup: dict[str, dict[str, int]],
    train_inference_frame: pd.DataFrame,
    segment_master: pd.DataFrame,
    segment_embedding_df: pd.DataFrame,
    segment_neighbor_features: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    cluster_profile_df: pd.DataFrame,
    segment_heterogeneity_df: pd.DataFrame,
    heterogeneity_config: dict[str, Any],
    segment_band_df: pd.DataFrame,
    comparable_index: dict[str, Any],
    location_artifacts: dict[str, Any],
    feature_bundle: dict[str, Any],
    access_bundle: dict[str, Any],
    model: HistGradientBoostingRegressor,
    preprocessor: ColumnTransformer,
    cluster_model: KMeans,
    spectral_cols: list[str],
    phase_logger: PhaseLogger,
) -> None:
    artifact_dir = ensure_dir(config.artifacts_dir)
    (artifact_dir / "config.json").write_text(json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "column_config.json").write_text(
        json.dumps(
            {
                "core_feature_cols": CORE_FEATURE_COLS,
                "access_feature_cols": ACCESS_FEATURE_COLS,
                "position_feature_cols": POSITION_FEATURE_COLS,
                "comparable_feature_cols": COMPARABLE_FEATURE_COLS,
                "location_feature_cols": location_artifacts["feature_names"],
                "spectral_cols": spectral_cols,
                "feature_layout": feature_layout,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "canonical_lookup.json").write_text(json.dumps(canonical_lookup, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "frequency_lookup.json").write_text(json.dumps(frequency_lookup, ensure_ascii=False, indent=2), encoding="utf-8")
    train_inference_frame.to_csv(artifact_dir / "train_inference_frame.csv", index=False)
    segment_master.to_csv(artifact_dir / "segment_master.csv", index=False)
    segment_embedding_df.to_csv(artifact_dir / "segment_embedding.csv", index=False)
    segment_embedding_df[["segment_key", "cluster", "cluster_confidence", "train_sample_count"]].to_csv(
        artifact_dir / "segment_cluster.csv", index=False
    )
    segment_neighbor_features.to_csv(artifact_dir / "segment_neighbor_features.csv", index=False)
    neighbor_df.to_csv(artifact_dir / "all_segment_neighbors.csv", index=False)
    cluster_profile_df.to_csv(artifact_dir / "cluster_profile.csv", index=False)
    segment_heterogeneity_df.to_csv(artifact_dir / "segment_heterogeneity.csv", index=False)
    (artifact_dir / "segment_heterogeneity_config.json").write_text(
        json.dumps(heterogeneity_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    segment_band_df.to_csv(artifact_dir / "all_segment_total_value_band.csv", index=False)
    save_pickle(comparable_index, artifact_dir / "comparable_train_index.pkl")
    joblib.dump(location_artifacts["encoders"], artifact_dir / "location_encoders.joblib")
    joblib.dump(location_artifacts["svds"], artifact_dir / "location_svds.joblib")
    joblib.dump(location_artifacts["scalers"], artifact_dir / "location_scalers.joblib")
    joblib.dump(feature_bundle, artifact_dir / "feature_imputer_scaler.joblib")
    joblib.dump(access_bundle, artifact_dir / "access_imputer_scaler.joblib")
    joblib.dump(model, artifact_dir / "model_total_value.joblib")
    joblib.dump(preprocessor, artifact_dir / "model_preprocessor.joblib")
    joblib.dump(cluster_model, artifact_dir / "cluster_model.joblib")
    (artifact_dir / "spectral_config.json").write_text(
        json.dumps({"spectral_cols": spectral_cols, "spectral_dim": len(spectral_cols)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "alpha_config.json").write_text(json.dumps(config.alpha, ensure_ascii=False, indent=2), encoding="utf-8")
    phase_logger.add(
        phase="artifact_persistence",
        objective="Save all train-time artifacts so inference can run without re-fitting.",
        inference_guard="run_test must only load artifacts and transform/predict; it must not fit again.",
        status="DONE",
        outputs=[str(path.name) for path in artifact_dir.iterdir()],
    )


def load_artifacts(config: PipelineConfig) -> dict[str, Any]:
    artifact_dir = config.artifacts_dir
    location_feature_info = json.loads((artifact_dir / "column_config.json").read_text(encoding="utf-8"))
    location_specs = [
        ("quan_norm", "loc_quan", 2),
        ("phuong_norm", "loc_phuong", 4),
        ("pho_canonical_norm", "loc_pho", 16),
    ]
    location_artifacts = {
        "encoders": joblib.load(artifact_dir / "location_encoders.joblib"),
        "svds": joblib.load(artifact_dir / "location_svds.joblib"),
        "scalers": joblib.load(artifact_dir / "location_scalers.joblib"),
        "feature_names": location_feature_info["location_feature_cols"],
        "specs": location_specs,
    }
    return {
        "config": json.loads((artifact_dir / "config.json").read_text(encoding="utf-8")),
        "column_config": location_feature_info,
        "canonical_lookup": json.loads((artifact_dir / "canonical_lookup.json").read_text(encoding="utf-8")),
        "frequency_lookup": json.loads((artifact_dir / "frequency_lookup.json").read_text(encoding="utf-8")),
        "segment_master": pd.read_csv(artifact_dir / "segment_master.csv"),
        "train_inference_frame": pd.read_csv(artifact_dir / "train_inference_frame.csv"),
        "segment_embedding": pd.read_csv(artifact_dir / "segment_embedding.csv"),
        "segment_neighbor_features": pd.read_csv(artifact_dir / "segment_neighbor_features.csv"),
        "neighbor_df": pd.read_csv(artifact_dir / "all_segment_neighbors.csv"),
        "cluster_profile": pd.read_csv(artifact_dir / "cluster_profile.csv"),
        "segment_heterogeneity": pd.read_csv(artifact_dir / "segment_heterogeneity.csv"),
        "segment_heterogeneity_config": json.loads(
            (artifact_dir / "segment_heterogeneity_config.json").read_text(encoding="utf-8")
        ),
        "segment_band_df": pd.read_csv(artifact_dir / "all_segment_total_value_band.csv"),
        "comparable_index": load_pickle(artifact_dir / "comparable_train_index.pkl"),
        "location_artifacts": location_artifacts,
        "feature_bundle": joblib.load(artifact_dir / "feature_imputer_scaler.joblib"),
        "access_bundle": joblib.load(artifact_dir / "access_imputer_scaler.joblib"),
        "model": joblib.load(artifact_dir / "model_total_value.joblib"),
        "preprocessor": joblib.load(artifact_dir / "model_preprocessor.joblib"),
        "cluster_model": joblib.load(artifact_dir / "cluster_model.joblib"),
        "spectral_cols": location_feature_info["spectral_cols"],
        "feature_layout": location_feature_info["feature_layout"],
    }


def stringify_osm_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(item) for item in value if str(item).strip())
    if pd.isna(value):
        return ""
    return str(value)


def standardize_roads_gdf(roads_gdf: Any) -> Any:
    roads = roads_gdf.copy()
    if roads.crs is None:
        roads = roads.set_crs(epsg=4326)
    if str(roads.crs).lower() != "epsg:4326":
        roads = roads.to_crs(epsg=4326)
    candidate_name_cols = ["road_name", "name", "street", "full_name"]
    candidate_id_cols = ["road_osm_id", "osmid", "osm_id", "id"]
    name_col = next((col for col in candidate_name_cols if col in roads.columns), None)
    id_col = next((col for col in candidate_id_cols if col in roads.columns), None)
    roads["road_name"] = roads[name_col].map(stringify_osm_value) if name_col else ""
    roads["road_osm_id"] = roads[id_col].map(stringify_osm_value) if id_col else roads.index.astype(str)
    roads = roads[roads.geometry.notna()].copy()
    roads = roads[~roads.geometry.is_empty].copy()
    return roads


def fetch_roads_for_point(lat: float, lon: float, config: PipelineConfig, cache: dict[str, Any]) -> Any:
    import geopandas as gpd
    import osmnx as ox

    if config.osm_roads_path:
        key = f"local::{config.osm_roads_path}"
        if key not in cache:
            cache[key] = standardize_roads_gdf(gpd.read_file(config.osm_roads_path))
        return cache[key]

    key = f"point::{round(lat, 6)}::{round(lon, 6)}"
    if key in cache:
        return cache[key]

    graph = ox.graph_from_point(
        (lat, lon),
        dist=config.evidence_fetch_dist_m,
        dist_type="bbox",
        network_type="all",
        simplify=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    cache[key] = standardize_roads_gdf(edges)
    return cache[key]


def choose_focus_row(test_df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    if config.evidence_focus_row_id:
        matched = test_df[test_df["row_id"] == config.evidence_focus_row_id]
        if not matched.empty:
            return matched.iloc[0]
    warned = test_df[test_df["warning_flag"] != "OK"]
    if not warned.empty:
        return warned.sort_values(["confidence_score", "row_id"]).iloc[0]
    return test_df.sort_values(["confidence_score", "row_id"]).iloc[0]


def build_radar_outputs(focus_row: pd.Series, focus_summary: pd.Series, out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir / "output_1")
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(out_dir / ".mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radar_data = {
        "focus_input_id": focus_row["row_id"],
        "segment_key": focus_row["segment_key"],
        "scores": {
            "geocode_score": 1.0 if focus_row.get("coord_warning_flag", "OK") == "OK" else 0.25,
            "spatial_neighbor_score": min(1.0, float(focus_row.get("neighbor_count_1000m", 0)) / 8.0),
            "comparable_count_score": min(1.0, float(focus_row.get("comparable_count", 0)) / 5.0),
            "note_match_score": float(focus_summary.get("mean_note_match_confidence", 0.0)),
            "road_evidence_score": float(focus_summary.get("road_evidence_confidence", 0.0)),
            "overall_confidence": float(focus_row.get("confidence_score", 0.0)),
        },
    }
    (out_dir / "evidence_radar_data.json").write_text(
        json.dumps(radar_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = list(radar_data["scores"].keys())
    values = list(radar_data["scores"].values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.plot(angles, values, color="#c34835", linewidth=2)
    ax.fill(angles, values, color="#c34835", alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(
        f"Evidence Radar: {focus_row['row_id']} | {focus_row.get('pho_canonical', '')}",
        fontsize=12,
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "evidence_radar.png", dpi=180)
    plt.close(fig)


def build_demo_map_html(
    focus_row: pd.Series,
    focus_roads_df: pd.DataFrame,
    test_df: pd.DataFrame,
    comparable_nodes_df: pd.DataFrame,
    config: PipelineConfig,
    out_dir: Path,
) -> None:
    import folium

    center = [float(focus_row["segment_lat"]), float(focus_row["segment_lon"])]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    folium.Marker(
        location=center,
        popup=folium.Popup(
            f"<b>{focus_row['row_id']}</b><br>{focus_row.get('pho_canonical','')}<br>"
            f"Pred: {format_money(focus_row.get('pred_total_value', np.nan))}<br>"
            f"Confidence: {focus_row.get('confidence_score', 0):.2f}",
            max_width=320,
        ),
        tooltip=f"Focus input: {focus_row['row_id']}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fmap)

    folium.Circle(
        radius=config.evidence_buffer_m,
        location=center,
        color="#c34835",
        fill=True,
        fill_opacity=0.08,
        weight=2,
        tooltip="Buffer 1km",
    ).add_to(fmap)

    sample_inputs = test_df.sort_values(["confidence_score", "row_id"]).head(config.evidence_map_input_limit)
    for row in sample_inputs.itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.segment_lat), float(row.segment_lon)],
            radius=5,
            color="#0f5c7a",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"{row.row_id} | {row.pho_canonical} | conf {row.confidence_score:.2f}",
        ).add_to(fmap)

    for road in focus_roads_df.itertuples(index=False):
        if not hasattr(road, "geometry_json") or not road.geometry_json:
            continue
        color = "#177245" if road.included_in_evidence else "#7a7a7a"
        dash_array = None if road.included_in_evidence else "6, 6"
        folium.GeoJson(
            road.geometry_json,
            style_function=lambda _feature, color=color, dash_array=dash_array: {
                "color": color,
                "weight": 4 if color == "#177245" else 2,
                "opacity": 0.9 if color == "#177245" else 0.5,
                "dashArray": dash_array,
            },
            tooltip=(
                f"{road.road_name or '(unnamed road)'} | len {road.clipped_highlighted_length_m:.1f}m | "
                f"match {road.note_match_method} | conf {road.note_match_confidence:.2f}"
            ),
        ).add_to(fmap)

    for row in comparable_nodes_df.itertuples(index=False):
        if pd.isna(row.segment_lat) or pd.isna(row.segment_lon):
            continue
        folium.CircleMarker(
            location=[float(row.segment_lat), float(row.segment_lon)],
            radius=4,
            color="#845ec2",
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"Comparable {row.row_id} | {row.pho_canonical} | "
                f"actual {format_money(row.tong_gia)}"
            ),
        ).add_to(fmap)

    fmap.save(str(out_dir / "demo_map.html"))


def build_osm_visual_evidence(
    test_df: pd.DataFrame,
    comparable_results_df: pd.DataFrame,
    train_reference_df: pd.DataFrame,
    config: PipelineConfig,
    out_dir: Path,
    phase_logger: PhaseLogger,
) -> pd.DataFrame:
    output_1_dir = ensure_dir(out_dir / "output_1")
    status = {
        "enabled": bool(config.enable_osm_evidence),
        "status": "SKIPPED",
        "reason": "OSM visual evidence is disabled in config.",
    }
    if not config.enable_osm_evidence:
        (output_1_dir / "osm_visual_evidence_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        phase_logger.add(
            phase="osm_visual_evidence",
            objective="Prepare optional OSM road evidence outputs only when local geometry is available.",
            inference_guard="OSM evidence must remain optional and must not block the tabular inference pipeline.",
            status=status["status"],
            notes=status["reason"],
            outputs=["osm_visual_evidence_status.json"],
        )
        return test_df

    try:
        import geopandas as gpd
        from shapely.geometry import Point, mapping

        valid_inputs = test_df[test_df["segment_lat"].notna() & test_df["segment_lon"].notna()].copy()
        if valid_inputs.empty:
            raise RuntimeError("No test rows have valid coordinates, so OSM visual evidence cannot be built.")

        roads_cache: dict[str, Any] = {}
        road_records: list[dict[str, Any]] = []
        segment_records: list[dict[str, Any]] = []

        for row in valid_inputs.itertuples(index=False):
            point_geom = Point(float(row.segment_lon), float(row.segment_lat))
            point_gdf = gpd.GeoDataFrame([{"row_id": row.row_id}], geometry=[point_geom], crs="EPSG:4326")
            point_metric = point_gdf.to_crs(epsg=3857).geometry.iloc[0]
            buffer_metric = point_metric.buffer(config.evidence_buffer_m)
            buffer_4326 = gpd.GeoSeries([buffer_metric], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

            roads_gdf = fetch_roads_for_point(float(row.segment_lat), float(row.segment_lon), config, roads_cache)
            roads_metric = roads_gdf.to_crs(epsg=3857).copy()
            clipped = roads_metric[roads_metric.intersects(buffer_metric)].copy()
            if clipped.empty:
                segment_records.append(
                    {
                        "input_id": row.row_id,
                        "segment_key": row.segment_key,
                        "note_road_name": row.pho_canonical,
                        "evidence_road_count": 0,
                        "excluded_road_count": 0,
                        "mean_note_match_confidence": 0.0,
                        "road_evidence_confidence": 0.0,
                        "buffer_radius_m": config.evidence_buffer_m,
                        "pred_total_value": row.pred_total_value,
                        "confidence_score": row.confidence_score,
                        "warning_flag": join_flags([row.warning_flag, "NO_ROAD_NOTE_MATCH"]),
                    }
                )
                continue

            clipped["geometry"] = clipped.geometry.intersection(buffer_metric)
            clipped = clipped[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
            clipped["clipped_highlighted_length_m"] = clipped.geometry.length
            clipped["distance_to_input_m"] = clipped.geometry.distance(point_metric)
            clipped["distance_to_note_m"] = clipped["distance_to_input_m"]
            clipped["note_segment_key"] = row.segment_key
            clipped["note_road_name"] = row.pho_canonical
            clipped["intersects_buffer_1km"] = True

            note_norm = normalize_text(row.pho_canonical)
            road_names_norm = clipped["road_name"].map(normalize_text)
            clipped["note_match_method"] = "NO_MATCH"
            clipped["note_match_confidence"] = 0.0
            exact_mask = road_names_norm == note_norm
            partial_mask = (~exact_mask) & road_names_norm.map(
                lambda value: bool(note_norm) and (note_norm in value or value in note_norm) if value else False
            )
            on_road_mask = (~exact_mask) & (~partial_mask) & (clipped["distance_to_note_m"] <= 15)
            near_road_mask = (~exact_mask) & (~partial_mask) & (~on_road_mask) & (clipped["distance_to_note_m"] <= 40)
            clipped.loc[exact_mask, ["note_match_method", "note_match_confidence"]] = ["NAME_EXACT", 1.0]
            clipped.loc[partial_mask, ["note_match_method", "note_match_confidence"]] = ["NAME_PARTIAL", 0.85]
            clipped.loc[on_road_mask, ["note_match_method", "note_match_confidence"]] = ["POINT_ON_ROAD", 0.8]
            clipped.loc[near_road_mask, ["note_match_method", "note_match_confidence"]] = ["ROAD_NEAR_NOTE", 0.6]
            clipped["note_match_status"] = clipped["note_match_confidence"] >= 0.6

            clipped["included_in_evidence"] = (
                clipped["intersects_buffer_1km"]
                & clipped["note_match_status"]
                & (clipped["clipped_highlighted_length_m"] <= config.max_clipped_road_length_m)
            )
            clipped["exclude_reason"] = np.where(
                clipped["clipped_highlighted_length_m"] > config.max_clipped_road_length_m,
                "EXCLUDED_CLIPPED_ROAD_TOO_LONG",
                np.where(
                    ~clipped["note_match_status"],
                    "NO_ROAD_NOTE_MATCH",
                    "",
                ),
            )
            clipped = clipped.to_crs(epsg=4326)
            clipped["geometry_wkt"] = clipped.geometry.to_wkt()
            clipped["geometry_json"] = clipped.geometry.apply(lambda geom: mapping(geom) if geom is not None else None)
            clipped["input_id"] = row.row_id

            included_count = int(clipped["included_in_evidence"].sum())
            excluded_count = int((~clipped["included_in_evidence"]).sum())
            mean_match_conf = float(clipped["note_match_confidence"].mean()) if len(clipped) else 0.0
            road_evidence_conf = float(
                np.clip(
                    0.45 * min(1.0, included_count / 5.0)
                    + 0.35 * mean_match_conf
                    + 0.20 * (1.0 if included_count > 0 else 0.0)
                    - 0.10 * (1.0 if excluded_count > included_count else 0.0),
                    0,
                    1,
                )
            )
            segment_records.append(
                {
                    "input_id": row.row_id,
                    "segment_key": row.segment_key,
                    "note_road_name": row.pho_canonical,
                    "input_lat": row.segment_lat,
                    "input_lon": row.segment_lon,
                    "buffer_radius_m": config.evidence_buffer_m,
                    "evidence_road_count": included_count,
                    "excluded_road_count": excluded_count,
                    "mean_note_match_confidence": mean_match_conf,
                    "road_evidence_confidence": road_evidence_conf,
                    "pred_total_value": row.pred_total_value,
                    "confidence_score": row.confidence_score,
                    "warning_flag": row.warning_flag,
                }
            )
            road_records.extend(clipped.to_dict(orient="records"))

        roads_df = pd.DataFrame(road_records)
        segments_df = pd.DataFrame(segment_records)
        if segments_df.empty:
            raise RuntimeError("No visual evidence segments were created from the current test rows.")

        focus_row = choose_focus_row(test_df, config)
        focus_summary = segments_df[segments_df["input_id"] == focus_row["row_id"]]
        if focus_summary.empty:
            focus_summary = segments_df.head(1)
        focus_summary_row = focus_summary.iloc[0]

        focus_roads_df = roads_df[roads_df["input_id"] == focus_row["row_id"]].copy()
        focus_comparable_ids = comparable_results_df[comparable_results_df["input_id"] == focus_row["row_id"]][
            "comparable_row_id"
        ].drop_duplicates()
        comparable_nodes_df = train_reference_df[train_reference_df["row_id"].isin(focus_comparable_ids)].copy()
        build_demo_map_html(focus_row, focus_roads_df, test_df, comparable_nodes_df, config, out_dir)
        build_radar_outputs(focus_row, focus_summary_row, out_dir)
        explanation_text = (
            f"Focus input: {focus_row['row_id']} | {focus_row.get('pho_canonical','')}\n"
            f"Segment key: {focus_row['segment_key']}\n"
            f"Predicted total value: {format_money(focus_row.get('pred_total_value', np.nan))}\n"
            f"Current confidence: {focus_row.get('confidence_score', 0):.2f}\n"
            f"Included evidence roads: {int(focus_summary_row.get('evidence_road_count', 0))}\n"
            f"Excluded evidence roads: {int(focus_summary_row.get('excluded_road_count', 0))}\n"
            f"Mean note match confidence: {focus_summary_row.get('mean_note_match_confidence', 0):.2f}\n"
            f"Road evidence confidence: {focus_summary_row.get('road_evidence_confidence', 0):.2f}\n"
        )
        (output_1_dir / "explanation_text.txt").write_text(explanation_text, encoding="utf-8")

        road_output_columns = [
            "input_id",
            "road_osm_id",
            "road_name",
            "note_segment_key",
            "note_road_name",
            "intersects_buffer_1km",
            "clipped_highlighted_length_m",
            "distance_to_input_m",
            "distance_to_note_m",
            "note_match_status",
            "note_match_method",
            "note_match_confidence",
            "included_in_evidence",
            "exclude_reason",
            "geometry_wkt",
        ]
        roads_save = (
            roads_df[road_output_columns].copy()
            if not roads_df.empty
            else pd.DataFrame(columns=road_output_columns)
        )
        roads_save.to_csv(output_1_dir / "visual_evidence_roads.csv", index=False)
        segments_df.to_csv(output_1_dir / "visual_evidence_segments.csv", index=False)

        updated_test = test_df.merge(
            segments_df[
                [
                    "input_id",
                    "evidence_road_count",
                    "excluded_road_count",
                    "mean_note_match_confidence",
                    "road_evidence_confidence",
                ]
            ].rename(columns={"input_id": "row_id"}),
            on="row_id",
            how="left",
        )
        updated_test["evidence_road_count"] = updated_test["evidence_road_count"].fillna(0).astype(int)
        updated_test["excluded_road_count"] = updated_test["excluded_road_count"].fillna(0).astype(int)
        updated_test["mean_note_match_confidence"] = updated_test["mean_note_match_confidence"].fillna(0.0)
        updated_test["road_evidence_confidence"] = updated_test["road_evidence_confidence"].fillna(0.0)

        status = {
            "enabled": True,
            "status": "DONE",
            "reason": "OSM road evidence was built successfully from local or fetched road geometry.",
            "focus_input_id": focus_row["row_id"],
            "road_rows": int(len(roads_save)),
            "segment_rows": int(len(segments_df)),
        }
        (output_1_dir / "osm_visual_evidence_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        phase_logger.add(
            phase="osm_visual_evidence",
            objective="Build buffer 1km road evidence, keep/exclude clipped roads, and render a demo map for the focus input.",
            inference_guard="OSM evidence is post-prediction explanation only; it must not leak back into train fitting unless explicitly enabled as a model feature in a separate experiment.",
            status="DONE",
            notes=status["reason"],
            outputs=[
                "visual_evidence_roads.csv",
                "visual_evidence_segments.csv",
                "demo_map.html",
                "evidence_radar_data.json",
                "evidence_radar.png",
                "explanation_text.txt",
                "osm_visual_evidence_status.json",
            ],
        )
        return updated_test
    except Exception as exc:
        status = {
            "enabled": True,
            "status": "FAILED",
            "reason": str(exc),
        }
        (output_1_dir / "osm_visual_evidence_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        phase_logger.add(
            phase="osm_visual_evidence",
            objective="Build buffer 1km road evidence, keep/exclude clipped roads, and render a demo map for the focus input.",
            inference_guard="OSM evidence is optional; if fetching or geometry processing fails, the main tabular inference outputs must still be valid.",
            status="FAILED",
            notes=str(exc),
            outputs=["osm_visual_evidence_status.json"],
        )
        return test_df


def save_schema_report(report: dict[str, Any], out_dir: Path) -> None:
    output_1_dir = out_dir / "output_1"
    output_1_dir.mkdir(parents=True, exist_ok=True)
    (output_1_dir / "input_schema_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_train_outputs(
    config: PipelineConfig,
    alias_report: pd.DataFrame,
    geo_report: pd.DataFrame,
    train_with_coordinates: pd.DataFrame,
    train_predictions: pd.DataFrame,
    train_oof_predictions: pd.DataFrame,
    all_clustered_data: pd.DataFrame,
    comparable_results: pd.DataFrame,
    segment_band_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    graph_config_report: pd.DataFrame,
    cluster_config_report: pd.DataFrame,
    workflow_metrics: dict[str, Any],
) -> None:
    out_dir = ensure_dir(config.output_dir)
    output_1_dir = ensure_dir(out_dir / "output_1")
    
    alias_report.to_csv(output_1_dir / "street_alias_report.csv", index=False)
    geo_report.to_csv(output_1_dir / "geo_quality_report.csv", index=False)
    train_with_coordinates.to_csv(output_1_dir / "train_with_coordinates.csv", index=False)
    train_predictions.to_csv(output_1_dir / "train_predictions.csv", index=False)
    train_oof_predictions.to_csv(output_1_dir / "train_oof_predictions.csv", index=False)
    all_clustered_data.to_csv(output_1_dir / "all_clustered_data.csv", index=False)
    comparable_results.to_csv(output_1_dir / "comparable_results.csv", index=False)
    segment_band_df.to_csv(output_1_dir / "all_segment_total_value_band.csv", index=False)
    ablation_df.to_csv(output_1_dir / "ablation_metrics.csv", index=False)
    graph_config_report.to_csv(output_1_dir / "graph_config_report.csv", index=False)
    cluster_config_report.to_csv(output_1_dir / "cluster_config_report.csv", index=False)
    (output_1_dir / "workflow_metrics.json").write_text(
        json.dumps(workflow_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_test_outputs(
    config: PipelineConfig,
    test_with_coordinates: pd.DataFrame,
    test_predictions: pd.DataFrame,
    comparable_results: pd.DataFrame,
) -> None:
    out_dir = ensure_dir(config.output_dir)
    output_1_dir = ensure_dir(out_dir / "output_1")
    
    # Gia_Test.csv stays in out_dir
    test_predictions.to_csv(out_dir / "Gia_Test.csv", index=False)
    
    # Other test files go to output_1
    test_with_coordinates.to_csv(output_1_dir / "test_with_coordinates.csv", index=False)
    if not comparable_results.empty:
        comparable_results.to_csv(output_1_dir / "comparable_results_test.csv", index=False)


def train_pipeline(config: PipelineConfig, phase_logger: PhaseLogger) -> dict[str, Any]:
    out_dir = ensure_dir(config.output_dir)
    raw_train = read_csv_file(config.train_file)
    raw_test = read_csv_file(config.test_file) if config.test_file.exists() else pd.DataFrame()
    raw_coord = read_csv_file(config.coord_file)
    schema_report = build_input_schema_report(raw_train, raw_test if not raw_test.empty else pd.DataFrame(columns=raw_train.columns), raw_coord)
    save_schema_report(schema_report, out_dir)
    if schema_report["schema_status"] != "OK":
        raise ValueError(f"Input schema invalid: {schema_report}")
    phase_logger.add(
        phase="schema_validation",
        objective="Validate standardized schema before any feature engineering or fitting.",
        inference_guard="Column assumptions must be explicit so train and test share the same normalized schema.",
        status="DONE",
        outputs=["input_schema_report.json"],
    )

    canonical_lookup = build_canonical_lookup(raw_train, raw_coord)
    alias_report = build_alias_report(canonical_lookup, raw_train, raw_coord, raw_test if not raw_test.empty else raw_train.head(0))
    train = add_basic_columns(raw_train, "train", canonical_lookup)
    coord = prepare_coordinate_dictionary(raw_coord, canonical_lookup)
    segment_master = build_segment_master(train, coord)
    train = attach_segment_info(train, segment_master, is_train=True)
    frequency_lookup = build_frequency_lookup(train)
    train = add_engineered_features(train, frequency_lookup)
    clean_train_for_diag = train[train["tong_gia"] > 0].copy().reset_index(drop=True)
    segment_heterogeneity_df, heterogeneity_config = build_segment_heterogeneity_table(clean_train_for_diag)
    heterogeneity_templates = build_segment_heterogeneity_templates(segment_heterogeneity_df, heterogeneity_config)
    train = attach_segment_heterogeneity(train, segment_heterogeneity_df, heterogeneity_templates)
    phase_logger.add(
        phase="data_readiness",
        objective="Build canonical streets, segment_key, coordinates, warnings, and engineered numeric features.",
        inference_guard="Only train data can define canonical lookup and train frequency statistics used later at inference.",
        status="DONE",
        outputs=["street_alias_report.csv", "geo_quality_report.csv"],
    )

    train, location_artifacts = fit_location_artifacts(train, config.random_state)
    feature_bundle = fit_scaled_bundle(train, CORE_FEATURE_COLS)
    access_bundle = fit_scaled_bundle(train, ACCESS_FEATURE_COLS)
    train, cluster_model, spectral_cols, cluster_config_report, graph_config_report = build_hybrid_embedding(
        train,
        location_artifacts["feature_names"],
        feature_bundle,
        access_bundle,
        config,
    )
    cluster_profile_df = build_cluster_profile_table(train)
    neighbor_df = build_segment_neighbors(segment_master, config)
    segment_neighbor_features = build_segment_neighbor_features(neighbor_df)
    train = attach_neighbor_features(train, segment_neighbor_features)
    train = add_cluster_profile_columns(train, cluster_profile_df)
    train = add_special_asset_columns(train)
    segment_embedding_df = build_segment_embedding_table(train, spectral_cols)
    train["segment_found_in_train"] = True
    train["assigned_cluster_method"] = "SEGMENT_LOOKUP"
    train["case_type"] = "CASE_1_SEGMENT_IN_TRAIN"
    train["cluster"] = train["assigned_cluster"]
    phase_logger.add(
        phase="hybrid_graph_embedding",
        objective="Fit location + position hybrid graph, spectral embedding, row clusters and segment embeddings.",
        inference_guard="Embedding and cluster are fit only on train; inference must reuse saved segment representations or fall back by KNN/location.",
        status="DONE",
        outputs=["all_clustered_data.csv", "segment_embedding.csv"],
    )

    clean_train = train[train["tong_gia"] > 0].copy().reset_index(drop=True)
    comparable_index = build_comparable_index(clean_train, segment_embedding_df, feature_bundle, access_bundle, config)
    comparable_results_train, comparable_summary_train = build_comparable_features(train, comparable_index, config, exclude_self=True)
    train = train.merge(comparable_summary_train, on="row_id", how="left")
    segment_band_df = build_segment_total_value_band(clean_train)
    train = add_segment_band_columns(train, segment_band_df)
    train = add_special_asset_columns(train)
    phase_logger.add(
        phase="comparable_train_only",
        objective="Build train-only comparable index and total-value bands for final training artifacts.",
        inference_guard="Comparable and price band must be derived from train-only candidates; test rows never participate in band creation.",
        status="DONE",
        outputs=["all_segment_total_value_band.csv", "comparable_results.csv"],
    )

    feature_layout = model_feature_layout(location_artifacts["feature_names"], spectral_cols)
    train_oof_predictions = build_oof_predictions(
        train,
        segment_embedding_df,
        feature_bundle,
        access_bundle,
        feature_layout,
        config,
    )
    train_oof_predictions = add_confidence_and_explanations(train_oof_predictions)
    phase_logger.add(
        phase="oof_evaluation",
        objective="Generate OOF predictions using fold-specific comparable features to reduce leakage.",
        inference_guard="Each validation fold must receive comparable/price-band features built only from its fit fold, not from all train rows.",
        status="DONE",
        outputs=["train_oof_predictions.csv"],
    )

    numeric_features = feature_layout["sets"]["model_5_comparable"]
    categorical_features = feature_layout["categorical_features"]
    model, preprocessor, train_pred_full = fit_model_and_predict(
        train,
        train,
        numeric_features,
        categorical_features,
        config.random_state,
    )
    train["pred_total_value"] = train_pred_full
    train["prediction_source"] = "full_model_prediction"
    train = add_confidence_and_explanations(train)
    phase_logger.add(
        phase="final_model_fit",
        objective="Fit the final log1p(Tổng giá trị) model on all valid train rows and persist inference artifacts.",
        inference_guard="The final model may fit on all clean train rows, but test inference must only load this saved model and transform new rows.",
        status="DONE",
        outputs=["model_total_value.joblib", "model_preprocessor.joblib"],
    )

    train_eval = train[train["actual_total_value"].notna()].copy()
    ablation_df = run_ablation(train, train_eval, feature_layout, config)
    all_clustered_data = train[
        [
            "row_id",
            "dataset_name",
            "segment_key",
            *spectral_cols,
            "assigned_cluster",
            "cluster_confidence",
        ]
    ].rename(columns={"assigned_cluster": "cluster"})

    geo_report = segment_master[
        [
            "segment_key",
            "pho",
            "sample_count",
            "train_sample_count",
            "segment_lat",
            "segment_lon",
            "coord_warning_flag",
            "distance_to_ward_median_m",
            "geo_note",
        ]
    ].copy()

    workflow_metrics = {
        "coord_match_rate_train": float((train["coord_match_status"] == "MATCHED_MANUAL_COORD").mean()),
        "train_rows_total": int(len(train)),
        "train_rows_positive_target": int((train["tong_gia"] > 0).sum()),
        "geo_outlier_segments": int(segment_master["coord_warning_flag"].str.contains("WARD_COORD_OUTLIER", na=False).sum()),
        "missing_geo_segments": int(segment_master["coord_warning_flag"].str.contains("MISSING_SEGMENT_GEO", na=False).sum()),
        "heterogeneous_segment_high_count": int(
            (segment_heterogeneity_df["segment_heterogeneity_flag"] == "HETEROGENEOUS_HIGH").sum()
        ),
        "heterogeneous_segment_medium_count": int(
            (segment_heterogeneity_df["segment_heterogeneity_flag"] == "HETEROGENEOUS_MEDIUM").sum()
        ),
    }
    workflow_metrics.update(
        prefix_metrics(
            metrics_dict(train_oof_predictions["actual_total_value"], train_oof_predictions["pred_total_value"]),
            "train_oof_",
        )
    )

    save_artifacts(
        config,
        canonical_lookup,
        feature_layout,
        frequency_lookup,
        train,
        segment_master,
        segment_embedding_df,
        segment_neighbor_features,
        neighbor_df,
        cluster_profile_df,
        segment_heterogeneity_df,
        heterogeneity_config,
        segment_band_df,
        comparable_index,
        location_artifacts,
        feature_bundle,
        access_bundle,
        model,
        preprocessor,
        cluster_model,
        spectral_cols,
        phase_logger,
    )
    save_train_outputs(
        config,
        alias_report,
        geo_report,
        train,
        train,
        train_oof_predictions,
        all_clustered_data,
        comparable_results_train,
        segment_band_df,
        ablation_df,
        graph_config_report,
        cluster_config_report,
        workflow_metrics,
    )

    build_diagnostic_reports(train_oof_predictions, out_dir)

    return {
        "train": train,
        "segment_embedding_df": segment_embedding_df,
        "segment_neighbor_features": segment_neighbor_features,
        "spectral_cols": spectral_cols,
        "feature_layout": feature_layout,
        "workflow_metrics": workflow_metrics,
    }


def inference_pipeline(config: PipelineConfig, phase_logger: PhaseLogger) -> dict[str, Any]:
    artifacts = load_artifacts(config)
    out_dir = ensure_dir(config.output_dir)
    raw_test = read_csv_file(config.test_file)
    test_has_target = "tong_gia" in raw_test.columns and raw_test["tong_gia"].notna().any()
    schema_report = {
        "train_missing_columns": [],
        "test_missing_columns": validate_required_columns(
            raw_test,
            ["quan", "phuong", "pho", "dien_tich_m2", "mat_tien_m", "chieu_dai_m", "so_mat_tien_tiep_giap", "khoang_cach_den_duong_chinh_m", "do_rong_ngo_nho_nhat_m", "lt_chuan"],
            "test",
        ),
        "coord_missing_columns": [],
        "test_has_target": bool(test_has_target),
        "schema_status": "OK",
    }
    save_schema_report(schema_report, out_dir)

    test = add_basic_columns(raw_test, "test", artifacts["canonical_lookup"])
    test = attach_segment_info(test, artifacts["segment_master"], is_train=False)
    test = add_engineered_features(test, artifacts["frequency_lookup"])
    heterogeneity_templates = build_segment_heterogeneity_templates(
        artifacts["segment_heterogeneity"],
        artifacts["segment_heterogeneity_config"],
    )
    test = attach_segment_heterogeneity(test, artifacts["segment_heterogeneity"], heterogeneity_templates)
    test = transform_location_features(test, artifacts["location_artifacts"])
    fallback_templates = build_location_fallback_templates(
        artifacts["segment_embedding"],
        artifacts["spectral_cols"],
        ["cluster_confidence", "cluster_center_distance_raw", "distance_to_cluster_center", "local_embedding_density"],
    )
    test = assign_inference_representation(
        test,
        artifacts["segment_embedding"],
        artifacts["segment_neighbor_features"],
        artifacts["spectral_cols"],
        fallback_templates,
        artifacts["cluster_model"],
        config,
    )
    test = add_cluster_profile_columns(test, artifacts["cluster_profile"])
    test = add_special_asset_columns(test)
    comparable_results_test, comparable_summary_test = build_comparable_features(
        test,
        artifacts["comparable_index"],
        config,
        exclude_self=False,
    )
    test = test.merge(comparable_summary_test, on="row_id", how="left")
    test = add_segment_band_columns(test, artifacts["segment_band_df"])
    test = add_special_asset_columns(test)
    phase_logger.add(
        phase="test_inference_preparation",
        objective="Load train artifacts, normalize test rows, assign case/cluster, and derive train-only comparable features.",
        inference_guard="Test rows can only transform with saved artifacts; they must not influence train bands, clusters, encoders or scalers.",
        status="DONE",
        outputs=["test_with_coordinates.csv", "comparable_results_test.csv"],
    )

    numeric_features = artifacts["feature_layout"]["sets"]["model_5_comparable"]
    categorical_features = artifacts["feature_layout"]["categorical_features"]
    X_test = artifacts["preprocessor"].transform(test[numeric_features + categorical_features])
    test["pred_total_value"] = np.expm1(artifacts["model"].predict(X_test))
    test["prediction_source"] = "full_model_prediction"
    test = add_confidence_and_explanations(test)
    test = build_osm_visual_evidence(
        test,
        comparable_results_test,
        artifacts["train_inference_frame"],
        config,
        out_dir,
        phase_logger,
    )
    test = add_confidence_and_explanations(test)
    if test_has_target:
        test["actual_total_value"] = test["tong_gia"]
        build_diagnostic_reports(test, out_dir)
        phase_logger.add(
            phase="test_metrics",
            objective="Compute evaluation metrics when test labels are available.",
            inference_guard="Metrics are optional at inference time and only computed when true labels exist.",
            status="DONE",
            outputs=[
                "test_metrics.csv",
                "error_by_year.csv",
                "error_by_segment.csv",
                "error_by_cluster.csv",
                "error_by_access_band.csv",
                "error_by_heterogeneous_segment.csv",
                "error_by_confidence_group.csv",
                "error_by_case.csv",
                "confidence_calibration_report.csv",
                "comparable_quality_report.csv",
            ],
        )
    # Tạo file kiem_dinh.csv gồm file data test + cột pred_total_value
    kiem_dinh = raw_test.copy()
    kiem_dinh["pred_total_value"] = test["pred_total_value"]
    kiem_dinh.to_csv(out_dir / "kiem_dinh.csv", index=False)
    
    save_test_outputs(config, test, test, comparable_results_test)

    if test_has_target:
        ablation_df = run_ablation(
            artifacts["train_inference_frame"],
            test,
            artifacts["feature_layout"],
            config,
        )
        ablation_df.to_csv(out_dir / "output_1" / "ablation_metrics.csv", index=False)

    workflow_metrics = {
        "coord_match_rate_test": float((test["coord_match_status"] == "MATCHED_MANUAL_COORD").mean()),
        "test_rows_total": int(len(test)),
        "test_has_target": bool(test_has_target),
    }
    if test_has_target:
        workflow_metrics.update(
            prefix_metrics(metrics_dict(test["actual_total_value"], test["pred_total_value"]), "test_")
        )
    output_1_dir = ensure_dir(out_dir / "output_1")
    existing_metrics_path = output_1_dir / "workflow_metrics.json"
    if existing_metrics_path.exists():
        try:
            existing_metrics = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
            existing_metrics.update(workflow_metrics)
            workflow_metrics = existing_metrics
        except Exception:
            pass
    (output_1_dir / "workflow_metrics.json").write_text(
        json.dumps(workflow_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"test": test, "workflow_metrics": workflow_metrics}


def main() -> None:
    config = parse_args()
    ensure_dir(config.output_dir)
    ensure_dir(config.artifacts_dir)
    phase_logger = PhaseLogger()
    if config.mode == "train":
        train_pipeline(config, phase_logger)
    elif config.mode == "test":
        inference_pipeline(config, phase_logger)
    else:
        train_pipeline(config, phase_logger)
        inference_pipeline(config, phase_logger)
    phase_logger.save(config.output_dir)


if __name__ == "__main__":
    main()
