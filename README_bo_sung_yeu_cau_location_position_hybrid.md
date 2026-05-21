# README — Yêu cầu bổ sung để hoàn thiện `location_position_hybrid_bhh.py`

**Tên tài liệu:** `README_bo_sung_yeu_cau_location_position_hybrid`  
**Mục tiêu:** bổ sung các yêu cầu còn thiếu để code `location_position_hybrid_bhh.py` bám sát README định hướng `Location + Vị trí thực tế Hybrid Workflow`.  
**Trạng thái hiện tại:** code đã có lõi location + vị trí thực tế + graph + embedding + comparable + model Tổng giá trị, nhưng chưa hoàn thiện đầy đủ phần inference chuẩn, chống leakage OOF, ablation, test không nhãn và OSM visual evidence.  
**Target chính:** `Tổng giá trị`  
**Target train:** `y = log1p(Tổng giá trị)`  
**Output chính:** `pred_total_value`

---

## 1. Tóm tắt mục tiêu bổ sung

Code hiện tại đã làm được nhiều phần lõi:

```text
- đọc train/test/coord;
- chuẩn hóa tên đường, alias, segment_key;
- parse tọa độ manual_lat/manual_lon;
- tạo segment_master và geo_quality_report;
- tạo feature lõi và access feature;
- tạo location embedding bằng OneHot/SVD;
- tạo hybrid graph gồm feature_graph + access_graph + location_graph + spatial_graph;
- fit spectral embedding;
- cluster;
- tạo neighbor 300m/500m/1000m/1500m/2000m;
- tạo comparable theo Tổng giá trị;
- train model log1p(Tổng giá trị);
- xuất train_predictions.csv và test_predictions.csv.
```

Tuy nhiên để đạt đúng workflow định hướng, cần bổ sung:

```text
1. Chuẩn hóa input schema linh hoạt hơn cho file tọa độ.
2. Tách train pipeline và test/inference pipeline.
3. Lưu/load artifact bằng joblib/json.
4. Cho phép test không có Tổng giá trị thật.
5. Sửa OOF comparable/price band để tránh leakage khi đánh giá.
6. Gán embedding/cluster cho segment test mới bằng KNN weighted.
7. Thêm ablation để chứng minh location + vị trí thực tế cải thiện.
8. Thêm metric đánh giá chi tiết: MAPE, theo năm, segment, cluster, confidence, case.
9. Bổ sung OSM visual evidence demo: buffer 1km, clipped roads, note check, loại đoạn cắt >3km.
10. Tạo output/README/log rõ ràng để nghiệm thu từng phase.
```

---

## 2. Yêu cầu 1 — Chuẩn hóa schema input linh hoạt hơn

### 2.1 Vấn đề

README định hướng file tọa độ có thể có các cột:

```text
district
ward
street
manual_lat
manual_lon
```

Nhưng code hiện tại chủ yếu dựa vào các tên đã được map sang:

```text
quan
phuong
pho
manual_lat
manual_lon
```

Nếu file tọa độ vẫn giữ tên gốc `district`, `ward`, `street`, code có nguy cơ không đọc đúng.

### 2.2 Yêu cầu bổ sung

Trong `COLUMN_MAP`, bổ sung:

```python
"district": "quan",
"ward": "phuong",
"street": "pho",
"lat": "manual_lat",
"lon": "manual_lon",
"latitude": "manual_lat",
"longitude": "manual_lon",
```

### 2.3 Kiểm tra bắt buộc sau đọc file

Tạo hàm:

```python
def validate_required_columns(df, required_cols, dataset_name):
    ...
```

Bắt buộc kiểm tra:

#### Train

```text
quan
phuong
pho hoặc duong_pho_clean
tong_gia
dien_tich_m2
mat_tien_m
chieu_dai_m
so_mat_tien_tiep_giap
khoang_cach_den_duong_chinh_m
do_rong_ngo_nho_nhat_m
lt_chuan
```

#### Test

```text
quan
phuong
pho hoặc duong_pho_clean
dien_tich_m2
mat_tien_m
chieu_dai_m
so_mat_tien_tiep_giap
khoang_cach_den_duong_chinh_m
do_rong_ngo_nho_nhat_m
lt_chuan
```

`tong_gia` ở test là optional.

#### Coordinate file

```text
quan
phuong
pho
manual_lat
manual_lon
```

### 2.4 Output cần thêm

```text
input_schema_report.json
```

Nội dung:

```json
{
  "train_missing_columns": [],
  "test_missing_columns": [],
  "coord_missing_columns": [],
  "test_has_target": true,
  "schema_status": "OK"
}
```

---

## 3. Yêu cầu 2 — Tách rõ Train Pipeline và Test/Inference Pipeline

### 3.1 Vấn đề

Code hiện tại chạy train và test liền trong `main()`.

Dùng cho demo một lượt thì được, nhưng chưa đúng chuẩn inference vì test đang nằm trong cùng tiến trình fit nhiều artifact.

### 3.2 Mục tiêu

Tách thành:

```text
run_train()
    học từ train
    fit scaler/encoder/location embedding/graph/embedding/cluster/model
    tạo comparable/price band train-only
    lưu artifact

run_test()
    đọc test
    load artifact train
    chuẩn hóa giống train
    transform test
    gán cluster/embedding
    tìm comparable train-only
    predict pred_total_value
    xuất test_predictions.csv
```

### 3.3 Cấu trúc đề xuất

```python
def run_train(config):
    raw_train = read_train(config.TRAIN_PATH)
    raw_coord = read_coord(config.COORD_PATH)

    train_processed, artifacts = fit_train_pipeline(raw_train, raw_coord, config)
    save_artifacts(artifacts, config.ARTIFACT_DIR)
    save_train_outputs(train_processed, artifacts, config.OUT_DIR)


def run_test(config):
    raw_test = read_test(config.TEST_PATH)
    artifacts = load_artifacts(config.ARTIFACT_DIR)

    test_pred = inference_pipeline(raw_test, artifacts, config)
    save_test_outputs(test_pred, config.OUT_DIR)


if __name__ == "__main__":
    if MODE == "train":
        run_train(config)
    elif MODE == "test":
        run_test(config)
    elif MODE == "train_test":
        run_train(config)
        run_test(config)
```

### 3.4 Tuyệt đối không làm trong `run_test()`

```text
- Không fit lại scaler.
- Không fit lại OneHotEncoder.
- Không fit lại TruncatedSVD.
- Không fit lại graph.
- Không fit lại SpectralEmbedding.
- Không fit lại KMeans cluster.
- Không train lại model.
- Không dùng giá test để tạo price band/comparable train-only.
```

---

## 4. Yêu cầu 3 — Lưu và load artifact đầy đủ

### 4.1 Artifact cần lưu sau train

Tạo thư mục:

```text
artifacts/
```

Bắt buộc lưu:

```text
artifacts/config.json
artifacts/column_config.json
artifacts/canonical_lookup.json
artifacts/segment_master.csv
artifacts/segment_embedding.csv
artifacts/segment_cluster.csv
artifacts/all_segment_neighbors.csv
artifacts/all_segment_total_value_band.csv
artifacts/comparable_train_index.parquet hoặc .pkl
artifacts/location_encoders.joblib
artifacts/location_svds.joblib
artifacts/location_scalers.joblib
artifacts/feature_imputer_scaler.joblib
artifacts/access_imputer_scaler.joblib
artifacts/model_total_value.joblib
artifacts/model_preprocessor.joblib
artifacts/cluster_model.joblib
artifacts/spectral_config.json
artifacts/alpha_config.json
```

### 4.2 Segment embedding artifact

File:

```text
segment_embedding.csv
```

Cột:

```text
segment_key
spectral_1
spectral_2
...
spectral_k
cluster
cluster_confidence
segment_lat_for_model
segment_lon_for_model
train_sample_count
```

### 4.3 Model artifact

Model đang dùng `HistGradientBoostingRegressor` với preprocessor. Cần lưu:

```python
joblib.dump(full_model, "model_total_value.joblib")
joblib.dump(fitted_preprocessor, "model_preprocessor.joblib")
```

---

## 5. Yêu cầu 4 — Test không có `Tổng giá trị` vẫn phải chạy được

### 5.1 Vấn đề

Code hiện tại tính test metrics trực tiếp bằng:

```python
test["tong_gia"]
```

Nếu test dùng để thẩm định thật và không có giá đúng, code sẽ không phù hợp.

### 5.2 Yêu cầu bổ sung

Tạo biến:

```python
test_has_target = "tong_gia" in test.columns and test["tong_gia"].notna().any()
```

Nếu `test_has_target = True`:

```text
- predict pred_total_value;
- tính actual_total_value;
- tính abs_error, ape, log_error;
- xuất test_metrics.csv.
```

Nếu `test_has_target = False`:

```text
- predict pred_total_value;
- không tính MAE/RMSE/MAPE;
- xuất price band, comparable, confidence, warning, explanation_text;
- actual_total_value để trống.
```

### 5.3 Output `test_predictions.csv`

Luôn có:

```text
row_id
segment_key
segment_found_in_train
assigned_cluster
assigned_cluster_method
segment_lat
segment_lon
neighbor_count_1000m
comparable_count
total_value_band_p25
total_value_band_median
total_value_band_p75
pred_total_value
confidence_score
warning_flag
explanation_text
```

Có thêm nếu test có giá thật:

```text
actual_total_value
abs_error
ape
log_error
```

---

## 6. Yêu cầu 5 — Sửa OOF comparable/price band để tránh leakage

### 6.1 Vấn đề

Hiện code tạo comparable và price band từ toàn bộ train trước khi chạy KFold OOF model.

Điều này có thể làm OOF metric lạc quan vì fold validation nhận feature price band được tính từ dữ liệu ngoài fold train.

### 6.2 Yêu cầu đúng

Khi đánh giá OOF:

```text
Với mỗi fold:
    train_fold = train[fit_idx]
    val_fold = train[val_idx]

    Fit comparable index chỉ từ train_fold.
    Tạo price band chỉ từ train_fold.
    Transform val_fold bằng comparable/price band từ train_fold.
    Train model trên train_fold.
    Predict val_fold.
```

Không dùng toàn bộ train để tạo feature price band cho fold validation.

### 6.3 Cách triển khai tối thiểu

Tạo hàm:

```python
def build_oof_comparable_features(train_df, folds, config):
    oof_features = []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        fold_train = train_df.iloc[fit_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        fold_comparable_index = fit_comparable_index(fold_train)
        fold_val_features = transform_comparable_features(fold_val, fold_comparable_index)
        fold_val_features["fold"] = fold
        oof_features.append(fold_val_features)

    return pd.concat(oof_features, ignore_index=True)
```

### 6.4 Final train vẫn dùng full train

Sau khi đánh giá OOF xong:

```text
Fit final comparable index bằng toàn bộ clean_train.
Fit final model bằng toàn bộ clean_train.
Dùng final artifacts cho test/inference thật.
```

---

## 7. Yêu cầu 6 — Gán cluster/embedding cho test segment mới bằng KNN weighted

### 7.1 Vấn đề

Hiện `assign_test_embedding()` chỉ merge theo `segment_key`.

Nếu segment test chưa có trong train:

```text
cluster = -1
spectral = 0
cluster_confidence = 0
```

Điều này chưa đúng mục tiêu Case 2.

### 7.2 Yêu cầu bổ sung

Nếu test segment chưa có embedding nhưng có tọa độ hợp lệ:

```text
1. Tìm các segment train gần nhất trong 1km.
2. Nếu thiếu thì mở 1.5km.
3. Nếu vẫn thiếu thì mở 2km.
4. Nếu vẫn thiếu thì lấy top K nearest train segments.
5. Gán spectral bằng trung bình có trọng số khoảng cách.
6. Gán cluster bằng majority vote có trọng số.
7. Ghi assigned_cluster_method = KNN_WEIGHTED_SPATIAL.
```

### 7.3 Công thức trọng số

```text
weight_i = exp(-distance_i / 700)
```

Embedding:

```text
test_spectral = sum(weight_i × train_segment_spectral_i) / sum(weight_i)
```

Cluster:

```text
assigned_cluster = cluster có tổng weight lớn nhất
```

Confidence:

```text
cluster_confidence = max_cluster_weight / total_weight
```

### 7.4 Warning

```text
LOW_NEIGHBOR_COUNT
FALLBACK_RADIUS_USED
FAR_FROM_KNOWN_SEGMENTS
OUT_OF_TRAIN_AREA
```

---

## 8. Yêu cầu 7 — Thêm Case 1 / Case 2 / Case 3 rõ ràng

### 8.1 Case 1 — Segment có trong train

```text
segment_found_in_train = true
assigned_cluster_method = SEGMENT_LOOKUP
```

Dùng trực tiếp:

```text
spectral_1..k
cluster
cluster_confidence
segment price band
neighbor features
```

### 8.2 Case 2 — Segment chưa có trong train nhưng có tọa độ

```text
segment_found_in_train = false
has_valid_coordinate = true
assigned_cluster_method = KNN_WEIGHTED_SPATIAL
```

Dùng:

```text
nearest train segments
spatial KNN
comparable trong 1km/1.5km/2km
```

### 8.3 Case 3 — Không có segment và không có tọa độ

```text
segment_found_in_train = false
has_valid_coordinate = false
assigned_cluster_method = LOCATION_FALLBACK
```

Fallback:

```text
phường
quận
location embedding
feature/access similarity
```

Warning:

```text
MISSING_SEGMENT_GEO
LOW_CONFIDENCE_PREDICTION
```

---

## 9. Yêu cầu 8 — Thêm ablation để chứng minh hiệu quả location + vị trí

### 9.1 Mục tiêu

Phải chứng minh:

```text
location + vị trí thực tế tốt hơn chỉ dùng location hoặc chỉ dùng feature gốc.
```

### 9.2 Các model cần chạy

```text
Model 1: core_property + access
Model 2: core_property + access + location_embedding
Model 3: core_property + access + location_embedding + segment_lat/lon + neighbor features
Model 4: core_property + access + location_embedding + street_spatial_graph embedding/cluster
Model 5: Model 4 + comparable total value band
Model 6: Model 5 + visual evidence features nếu đã có
Model 7: Model 5 + OSM POI nếu bật sau này
```

### 9.3 Output

```text
ablation_metrics.csv
```

Cột:

```text
model_name
feature_set
train_rows
test_rows
mae_total_value
rmse_total_value
mape_total_value
median_absolute_error_total_value
mean_log_error
case1_mae
case2_mae
high_confidence_mape
low_confidence_mape
```

### 9.4 Tiêu chí nghiệm thu

```text
Model có location + vị trí thực tế phải cải thiện MAE/MAPE hoặc tăng khả năng giải thích so với model chỉ location.
Nếu không cải thiện, phải giữ warning và xem lại chất lượng tọa độ/neighbor/comparable.
```

---

## 10. Yêu cầu 9 — Bổ sung metric đánh giá chi tiết

### 10.1 Metrics tổng thể

```text
MAE_total_value
RMSE_total_value
MAPE_total_value
Median_absolute_error_total_value
Mean_log_error
```

### 10.2 Metrics theo năm

```text
error_by_year.csv
```

Cột:

```text
nam
row_count
mae
rmse
mape
median_absolute_error
```

### 10.3 Metrics theo segment

```text
error_by_segment.csv
```

Cột:

```text
segment_key
pho_canonical
row_count
mae
mape
median_absolute_error
confidence_mean
```

### 10.4 Metrics theo cluster

```text
error_by_cluster.csv
```

Cột:

```text
cluster
row_count
mae
mape
median_absolute_error
confidence_mean
```

### 10.5 Metrics theo confidence group

```text
error_by_confidence_group.csv
```

Nhóm:

```text
high: confidence_score >= 0.8
medium_high: 0.6 <= confidence_score < 0.8
medium: 0.4 <= confidence_score < 0.6
low: confidence_score < 0.4
```

Kiểm tra:

```text
Nhóm confidence cao phải có sai số thấp hơn nhóm confidence thấp.
Nếu không, công thức confidence cần hiệu chỉnh.
```

### 10.6 Metrics theo case

```text
error_by_case.csv
```

Cột:

```text
case_type
row_count
mae
mape
median_absolute_error
```

Case:

```text
CASE_1_SEGMENT_IN_TRAIN
CASE_2_NEW_SEGMENT_WITH_COORD
CASE_3_LOCATION_FALLBACK
```

---

## 11. Yêu cầu 10 — OSM visual evidence demo

### 11.1 Trạng thái hiện tại

Code hiện tại chưa có nhánh này.

### 11.2 Mục tiêu

Tạo demo trực quan để giải thích giá:

```text
input node
→ buffer 1km
→ road geometry OSM
→ clip road giao với buffer
→ tô đậm phần đường nằm trong buffer
→ check đường tô đậm có liên hệ với note đã chấm không
→ loại đoạn cắt > 3km
→ hiển thị comparable nodes
→ tạo radar chart/evidence summary
```

### 11.3 Dependencies

```text
geopandas
shapely
pyproj
osmnx
folium
matplotlib hoặc plotly
```

### 11.4 Hàm cần có

```python
def build_osm_road_evidence(input_point, note_info, config):
    ...


def clip_roads_by_buffer(roads_gdf, buffer_geom):
    ...


def check_note_consistency(clipped_roads, note_point, note_road_name, segment_key):
    ...


def build_demo_map(input_point, buffer_geom, evidence_roads, comparable_nodes):
    ...


def build_evidence_radar(evidence_summary):
    ...
```

### 11.5 Quy tắc giữ/loại road

Giữ nếu:

```text
intersects_buffer_1km = true
note_match_status = true hoặc note_match_confidence đủ cao
clipped_highlighted_length_m <= 3000
```

Loại nếu:

```text
clipped_highlighted_length_m > 3000
```

Flag:

```text
EXCLUDED_CLIPPED_ROAD_TOO_LONG
NO_ROAD_NOTE_MATCH
LOW_ROAD_EVIDENCE_CONFIDENCE
```

### 11.6 Output

```text
visual_evidence_roads.csv
visual_evidence_segments.csv
demo_map.html
evidence_radar_data.json
evidence_radar.png
```

---

## 12. Yêu cầu 11 — Tách output train/test rõ hơn

### 12.1 Train output

```text
train_with_coordinates.csv
train_predictions.csv
train_oof_predictions.csv
```

`train_predictions.csv` nên phân biệt:

```text
prediction_source = oof_prediction hoặc full_model_prediction
```

### 12.2 Test output

```text
test_with_coordinates.csv
test_predictions.csv
test_metrics.csv nếu có target
```

### 12.3 Artifact output

```text
artifacts/
```

Dùng lại được cho inference thật.

### 12.4 Diagnostic output

```text
workflow_metrics.json
geo_quality_report.csv
street_alias_report.csv
ablation_metrics.csv
error_by_year.csv
error_by_segment.csv
error_by_cluster.csv
error_by_confidence_group.csv
error_by_case.csv
```

---

## 13. Yêu cầu 12 — Config hóa đường dẫn và tham số

### 13.1 Vấn đề

Code đang hard-code:

```python
BASE_DIR = Path("/Users/chudinhminh/Downloads/E")
```

### 13.2 Yêu cầu

Cho phép truyền qua:

```text
config.yaml
hoặc argparse
```

Ví dụ `config.yaml`:

```yaml
base_dir: /Users/chudinhminh/Downloads/E
train_path: DT_Train_binh_hung_hoa.csv
test_path: DT_Test_Phuong_Binh_Hung_Hoa.csv
coord_path: toa_do_Phuong_Binh_Hung_Hoa.csv
out_dir: output_location_position_hybrid
artifact_dir: artifacts
mode: train_test

random_state: 42
sigma_spatial_m: 700
neighbor_radii_m: [300, 500, 1000, 1500, 2000]
min_comparable_count: 5
top_k_comparable: 5
max_clipped_road_length_m: 3000

alpha:
  feature: 0.35
  access: 0.15
  location: 0.20
  spatial: 0.30
```

---

## 14. Thứ tự ưu tiên triển khai

### Ưu tiên 1 — Sửa để code chắc và đúng inference

```text
1. Bổ sung schema map district/ward/street.
2. Tách run_train() và run_test().
3. Lưu/load artifact.
4. Cho phép test không có Tổng giá trị.
5. Sửa metrics khi test không nhãn.
```

### Ưu tiên 2 — Sửa đánh giá cho sạch leakage

```text
6. Làm OOF comparable/price band theo fold.
7. Thêm MAPE và breakdown metrics.
8. Thêm error_by_year/segment/cluster/confidence/case.
```

### Ưu tiên 3 — Hoàn thiện Case 2

```text
9. Gán cluster/embedding cho test segment mới bằng KNN weighted.
10. Thêm assigned_cluster_method và case_type.
```

### Ưu tiên 4 — Chứng minh hiệu quả mô hình

```text
11. Thêm ablation_metrics.csv.
12. So sánh core vs location vs location+position vs comparable.
```

### Ưu tiên 5 — Demo trực quan OSM

```text
13. Thêm OSM road clipping.
14. Tạo visual_evidence_roads.csv.
15. Tạo demo_map.html.
16. Tạo evidence radar chart.
```

---

## 15. Checklist nghiệm thu sau khi sửa

```text
[ ] Code đọc được train/test/coord dù coord dùng district/ward/street.
[ ] Có validate_required_columns và input_schema_report.json.
[ ] Có run_train() và run_test() tách riêng.
[ ] Train lưu đủ artifact vào artifacts/.
[ ] Test load artifact và không fit lại.
[ ] Test không có Tổng giá trị vẫn predict được.
[ ] Nếu test có Tổng giá trị thì tính MAE/RMSE/MAPE.
[ ] OOF metrics không dùng comparable/price band từ toàn train.
[ ] Segment test mới được gán cluster bằng KNN weighted nếu có tọa độ.
[ ] Output có case_type và assigned_cluster_method.
[ ] Có ablation_metrics.csv.
[ ] Có error_by_year.csv.
[ ] Có error_by_segment.csv.
[ ] Có error_by_cluster.csv.
[ ] Có error_by_confidence_group.csv.
[ ] Có error_by_case.csv.
[ ] OSM visual evidence tạo được visual_evidence_roads.csv nếu bật.
[ ] OSM visual evidence loại clipped road >3km.
[ ] demo_map.html hiển thị input node, buffer 1km, đường tô đậm, comparable nodes.
[ ] README/output giải thích rõ train là học, test là inference.
```

---

## 16. Câu yêu cầu ngắn cho Codex/dev

```text
Hãy nâng cấp `location_position_hybrid_bhh.py` theo README bổ sung này.
Giữ nguyên target là `Tổng giá trị` và target train là `log1p(Tổng giá trị)`.
Không dùng Đơn giá làm target.
Tách rõ train và test/inference.
Train được phép fit scaler/encoder/graph/embedding/cluster/model và lưu artifact.
Test chỉ được load artifact, transform và predict, không fit lại.
Sửa OOF comparable/price band để tránh leakage.
Cho phép test không có Tổng giá trị thật.
Bổ sung KNN weighted để gán cluster/embedding cho segment test mới có tọa độ.
Bổ sung ablation và metrics chi tiết.
Sau cùng, nếu bật OSM visual evidence, tạo buffer 1km, clip road OSM, tô đậm đoạn đường nằm trong buffer, kiểm tra note, loại đoạn cắt >3km và xuất demo_map.html.
```

---

## 17. Kết luận

Sau khi bổ sung các yêu cầu trên, workflow sẽ đạt đúng định hướng:

```text
Location + vị trí thực tế
→ hybrid graph
→ spectral embedding/cluster
→ comparable Tổng giá trị train-only
→ model dự đoán log1p(Tổng giá trị)
→ test inference không fit lại
→ confidence/warning
→ nếu cần: OSM visual evidence chứng minh vùng giá.
```

Trạng thái mục tiêu sau khi hoàn thiện:

```text
- Dùng được cho demo model.
- Dùng được cho test có nhãn để đánh giá.
- Dùng được cho test không nhãn để thẩm định.
- Có artifact để chạy lại inference.
- Có output giải thích và cảnh báo.
- Có đường nâng cấp sang demo bản đồ OSM.
```
