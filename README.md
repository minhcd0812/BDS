[1.md](https://github.com/user-attachments/files/28196597/1.md)
# Workflow Chính — `location_position_hybrid_bhh_v2.py`

> Pipeline định giá bất động sản khu Phường Bình Hưng Hòa, Quận Bình Tân, TP.HCM  
> Chạy qua `run_all.py` → `main()` → `train_pipeline()` → `inference_pipeline()`

---

## Tổng quan kiến trúc

```
INPUT FILES                  TRAIN PIPELINE              INFERENCE PIPELINE          OUTPUT
──────────                   ──────────────              ──────────────────          ──────
DT_Train_...csv  ─┐                                                                test_1.html
DT_Test_...csv   ─┤─► main() ─► train_pipeline() ─► inference_pipeline() ─►       demo_map.html
toa_do_...csv    ─┘             (fit + save)          (load + predict)             Gia_Test.csv
                                     │                       │                     kiem_dinh.csv
                                 artifacts/              output_1/
```

---

## PHASE 0 — Khởi tạo & Đọc dữ liệu

| Bước | Hàm | Mô tả |
|------|-----|--------|
| 0.1 | `parse_args()` | Đọc tham số CLI / config file → `PipelineConfig` |
| 0.2 | `read_csv_file()` | Đọc 3 file CSV: train, test, coordinate |
| 0.3 | `build_canonical_lookup()` | Xây từ điển chuẩn hóa tên đường (normalize → canonical) |
| 0.4 | `build_alias_report()` | Báo cáo alias: tên nào là canonical, tên nào bị merge |
| 0.5 | `build_input_schema_report()` | Kiểm tra cột bắt buộc, phát hiện thiếu dữ liệu |

**Config quan trọng:**
- `alpha = {feature: 0.35, access: 0.15, location: 0.20, spatial: 0.30}` — trọng số similarity
- `kfold_splits = 5` — số fold OOF
- `spectral_dim = 6` — số chiều embedding spectral
- `top_k_comparable = 5` — số căn tương đồng

---

## PHASE 1 — Tiền xử lý & Tọa độ

```
train/test CSV
      │
      ▼
add_basic_columns()          ← Tạo row_id, segment_key, pho_canonical, pho_norm
      │
      ▼
build_segment_master()       ← Merge tọa độ thủ công với train counts
      │                         Kiểm tra outlier lat/lon, WARD_COORD_OUTLIER
      ▼
attach_segment_info()        ← Gán tọa độ segment vào từng hàng train/test
      │
      ▼
add_engineered_features()    ← Tính features phái sinh:
                                log_dien_tich, chi_so_hinhdang, chi_so_loithe,
                                ti_le_mat_tien, log_khoang_cach, log_do_rong_ngo,
                                ngo_x_lt, access_band (mat_tien/hem_rong/hem_nho...)
                                frequency (pho_rare, phuong_rare)
```

**Coordinate logic:**
- Tìm trong `toa_do_...csv` theo `segment_key`
- Nếu khớp → `MATCHED_MANUAL_COORD`
- Nếu không → `MISSING_SEGMENT_GEO` (fallback ward centroid)

---

## PHASE 2 — Segment Heterogeneity

```
build_segment_heterogeneity_table()
      │  Tính cho mỗi segment:
      │  - segment_price_cv (hệ số biến thiên giá)
      │  - segment_price_iqr_ratio
      │  - segment_heterogeneity_score
      ▼
attach_segment_heterogeneity()
      │  Gán flag:
      │  - STABLE_SEGMENT (score thấp)
      │  - HETEROGENEOUS_MEDIUM
      │  - HETEROGENEOUS_HIGH (phạt confidence)
      │  - LOW_SAMPLE_SEGMENT
      ▼
[Penalty áp vào confidence_score cuối]
```

---

## PHASE 3 — Hybrid Graph Embedding (Tim lõi mô hình)

```
build_hybrid_embedding()
      │
      ├─ [1] Location Embedding (SVD)
      │      OneHotEncode(quan, phuong, pho_canonical)
      │      TruncatedSVD → loc_quan_* (2D), loc_phuong_* (4D), loc_pho_* (16D)
      │      RobustScaler → chuẩn hóa
      │
      ├─ [2] Spectral Embedding
      │      Build kNN graph từ segment_key với kernel Gaussian
      │      alpha = {feature:0.35, access:0.15, location:0.20, spatial:0.30}
      │      SpectralEmbedding(n_components=6) → spectral_0..5
      │
      ├─ [3] KMeans Clustering
      │      Tìm k tối ưu (silhouette score) trong [4..12]
      │      min_cluster_size = 30
      │      Gán cluster, cluster_confidence cho mỗi hàng
      │
      └─ [4] Graph Config Report
             Ghi lại n_segments, sigma, alpha, silhouette score
```

**Output:** `spectral_0..5`, `assigned_cluster`, `cluster_confidence`

---

## PHASE 4 — Segment Embeddings & Neighbors

```
build_segment_embedding_table()   ← Trung bình spectral/cluster theo segment_key
      │
build_cluster_profile_table()     ← Thống kê giá theo cluster (p25/median/p75)
      │
build_segment_neighbors()         ← Haversine distance giữa tất cả segments
      │                              Radii: 300m, 500m, 1000m, 1500m, 2000m
      ▼
attach_neighbor_features()        ← neighbor_count_300m, 500m, 1000m
                                     nearest_segment_distance_m

add_special_asset_columns()       ← Phân loại: NORMAL / SPECIAL_MEDIUM / SPECIAL_HIGH
                                     Dựa trên: diện tích, mặt tiền, số mặt tiếp giáp
```

---

## PHASE 5 — Comparable Index & Features

```
build_comparable_index()
      │  Với mỗi train row: lưu {feature_vector, access_band, cluster, segment_key}
      │  alpha-weighted similarity = 0.35*feature + 0.15*access + 0.20*location + 0.30*spatial
      ▼
build_comparable_features()
      │  Tìm top-K (=5) căn tương đồng nhất (exclude_self=True khi train)
      │  Tính:
      │  - total_value_band_p25 / median / p75
      │  - comparable_count, comparable_confidence
      │  - same_cluster_ratio, access_band_match_ratio
      │  - band_confidence, mean_candidate_reliability
      │  - same_segment_ratio, heterogeneous_comparable_ratio
      │  - weighted_band_width_ratio
      │  Cảnh báo: LOW_COMPARABLE_COUNT, ACCESS_MISMATCH_POOL, WIDE_PRICE_BAND...
      ▼
build_segment_total_value_band()
      │  Tính band giá theo segment_key từ tất cả train rows
      ▼
add_segment_band_columns()        ← Merge band vào dataframe chính
```

---

## PHASE 6 — OOF Evaluation (5-Fold Cross Validation)

```
build_oof_predictions()
      │
      ▼  Với mỗi fold k=0..4:
      ├─ fold_train = train[fit_idx]
      ├─ fold_val   = train[val_idx]
      │
      ├─ Build comparable index CHỈ từ fold_train (chống leakage)
      ├─ Build comparable features cho fold_train và fold_val
      │
      ├─ Fit HistGradientBoostingRegressor trên fold_train
      │  (log1p target, 5 feature sets: core → location → position → embedding → comparable)
      │
      └─ Predict fold_val → fold_pred
      │
      ▼
Concat tất cả fold → train_oof_predictions
      │
add_confidence_and_explanations() ← Tính confidence_score, explanation_text
      │
build_diagnostic_reports()        ← error_by_segment, error_by_cluster, confidence_calibration...
```

**OOF Metrics (lần chạy hiện tại):**
- **MAPE = 7.0%** ✅
- MAE = 488.7M VNĐ
- Median AE = 240.2M VNĐ

---

## PHASE 7 — Final Model Fit & Artifact Save

```
fit_model_and_predict()
      │  Train toàn bộ clean_train (tong_gia > 0)
      │  Model: HistGradientBoostingRegressor
      │    learning_rate=0.05, max_depth=6, max_iter=350
      │    min_samples_leaf=12, l2_regularization=0.05
      │  Target: log1p(tong_gia) → expm1 để ra giá
      ▼
save_artifacts()
      │  artifacts/
      │  ├── model_total_value.joblib
      │  ├── model_preprocessor.joblib
      │  ├── cluster_model.joblib
      │  ├── segment_embedding.csv
      │  ├── comparable_train_index.pkl
      │  ├── location_encoders/svds/scalers.joblib
      │  ├── canonical_lookup.json
      │  ├── frequency_lookup.json
      │  └── config.json, column_config.json, alpha_config.json...
```

---

## PHASE 8 — Inference Pipeline (Test)

```
load_artifacts()              ← Nạp toàn bộ artifacts từ Phase 7

add_basic_columns(test)       ← Chuẩn hóa tên đường, tạo segment_key
attach_segment_info(test)     ← Gán tọa độ (CASE_1 / CASE_2 / CASE_3)
add_engineered_features(test) ← Các features phái sinh
attach_segment_heterogeneity()
transform_location_features() ← SVD location embedding (load saved encoders)

assign_inference_representation()
      │  CASE_1: segment có trong train → lookup trực tiếp
      │  CASE_2: segment mới + có tọa độ → KNN từ segment_embedding
      │  CASE_3: fallback location → dùng pho/phuong/quan embedding

build_comparable_features()   ← Tìm comparable từ train index (exclude_self=False)
add_segment_band_columns()    ← Merge price band từ train

preprocessor.transform(test)
model.predict(X_test)         ← Predict log1p → expm1
add_confidence_and_explanations()
build_osm_visual_evidence()   ← (nếu --enable-osm-evidence)
```

---

## PHASE 9 — OSM Visual Evidence (Optional)

```
(Kích hoạt bằng --enable-osm-evidence)

fetch_roads_for_point()       ← Tải đường phố từ OpenStreetMap (osmnx)
      │  Buffer 1000m xung quanh mỗi test point
      │  Clip, tính clipped_highlighted_length_m
      │  Match tên đường: NAME_EXACT / NAME_PARTIAL / POINT_ON_ROAD / ROAD_NEAR_NOTE
      ▼
build_demo_map_html()         ← demo_map.html (folium)
build_radar_outputs()         ← evidence_radar.png + evidence_radar_data.json
```

---

## PHASE 10 — Save Outputs

```
ouput_tai_lieu_lien_quan/
├── Gia_Test.csv                    ← Kết quả test với pred_total_value
├── kiem_dinh.csv                   ← Raw test + pred (để kiểm định)
├── test_1.html                     ← Bản đồ chính (highlight đường phố)
├── demo_map.html                   ← Bản đồ OSM evidence (folium)
├── highlightable_roads.js          ← Data JS cho test_1.html
├── train_points.js / test_points.js
└── output_1/
    ├── train_oof_predictions.csv
    ├── train_with_coordinates.csv
    ├── train_predictions.csv
    ├── comparable_results.csv / _test.csv
    ├── all_segment_total_value_band.csv
    ├── error_by_*.csv              ← Báo cáo lỗi chi tiết
    ├── confidence_calibration_report.csv
    ├── workflow_metrics.json
    ├── workflow_phase_log.json / .md
    ├── visual_evidence_roads.csv
    ├── evidence_radar.png
    └── osm_visual_evidence_status.json
```

---

## Công thức Confidence Score

```python
score = (
    0.18 * geocode_score          # tọa độ có hợp lệ không
  + 0.13 * location_score         # tên đường có đầy đủ không
  + 0.12 * spatial_score          # mật độ láng giềng 1000m
  + 0.18 * comparable_count_score # số căn tương đồng (max 5)
  + 0.09 * area_score             # diện tích có bị relaxed không
  + 0.10 * price_band_score       # độ hẹp của dải giá
  + 0.10 * cluster_score          # tỷ lệ cùng cluster
  + 0.08 * access_context_score   # tỷ lệ cùng access_band
  + 0.08 * band_confidence        # band confidence
  + 0.07 * reliability_score      # reliability của candidates
  + 0.10 * road_evidence_score    # (OSM evidence nếu bật)
) - penalty                       # case_penalty + warning_penalty + heterogeneity + special
```

---

## Feature Sets (5 Models Ablation)

| Model | Features |
|-------|---------|
| model_1_core_access | core + access + năm + rare |
| model_2_location | + SVD location (22D) |
| model_3_location_position | + lat/lon/neighbor counts |
| model_4_hybrid_embedding | + cluster + spectral (6D) |
| **model_5_comparable** ← **Dùng chính** | + comparable features (10D) |
