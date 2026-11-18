# BÁO CÁO DỰ ÁN: Qwen2.5-VL cho Detection Doctor (Thí nghiệm khả thi)

**Ngày báo cáo:** 18/11/2026  
**Thư mục làm việc:** `qwen-vlm-poc/`  
**Người thực hiện:** Cline (AI Coding Agent)

---

## 1. Mục tiêu dự án

Trả lời câu hỏi: **Có nên dùng Qwen2.5-VL-7B (FPT Cloud) làm thành phần giải thích lỗi của Detection Doctor không, và nên giao cho nó việc gì.**

Đây là nghiên cứu khả thi, không nằm trên đường chính của MVP 22/11. Mục tiêu là có **số đo thay vì cảm tính** trước khi quyết định tích hợp VLM vào hệ thống chính.

---

## 2. Môi trường & cài đặt

### 2.1 Dependencies đã cài

Tất cả các gói trong `requirements.txt` đã được cài thành công:

| Gói | Phiên bản | Vai trò |
|-----|-----------|---------|
| Pillow | 13.0.0 | Xử lý ảnh (PIL module) |
| requests | 2.34.2 | Gọi API HTTP |
| python-dotenv | 1.2.4 | Đọc file `.env` |
| huggingface_hub | 2.4.0 | Tải dataset từ HuggingFace |

Lỗi ban đầu `ModuleNotFoundError: No module named 'PIL'` đã được khắc phục bằng lệnh:
```
pip install -r requirements.txt
```

### 2.2 Biến môi trường

File `.env` đã được cấu hình với `FPT_API_KEY`. File này **KHÔNG được commit** vào git.

---

## 3. Cấu trúc dự án

```
qwen-vlm-poc/
├── README.md                 # Hướng dẫn và kết luận hiện tại
├── requirements.txt          # Dependencies
├── .env.example              # Mẫu biến môi trường
├── .env                      # API key (không commit)
├── main.py                   # Thí nghiệm cũ: phân tích cả cảnh
├── check_env.py              # Kiểm tra API key
├── eval/
│   ├── study.py              # Chạy thí nghiệm probe từng đối tượng
│   ├── cases.json            # 7 ca thử nghiệm ban đầu (đã verified)
│   ├── cases_bdd.json        # 60 ca từ BDD100K val
│   ├── evaluate.py           # So sánh output VLM với GT
│   ├── bdd100k_to_gt.py      # Script cũ
│   ├── gt_crop_test.json     # Ground truth mẫu
│   ├── gt_sample_output.json # Output mẫu
│   └── sample_bdd_labels.json# Dữ liệu mẫu tổng hợp
├── vlm/
│   ├── client.py             # Gọi API Qwen, retry
│   ├── parsing.py            # Parse JSON từ phản hồi VLM
│   ├── geometry.py           # Hình học xác định (touches_border, iou, crop...)
│   ├── probe.py              # Probe một đối tượng (crop + gọi API)
│   ├── metrics.py            # Chấm điểm: parse_rate, schema_rate, recall, false_alarm...
│   ├── scene_checks.py       # Kiểm tra mâu thuẫn nội tại
│   └── labels.py             # Chuẩn hóa tên lớp
├── prompts/
│   ├── object_probe_v1.txt   # Prompt hướng mới (probe từng đối tượng)
│   └── traffic_analysis*.txt # Prompt cũ (phân tích cả cảnh, giữ để so sánh)
├── tools/
│   ├── draw_boxes.py         # Vẽ bbox lên ảnh để kiểm tra nhãn bằng mắt
│   └── bdd_to_cases.py       # Chuyển nhãn BDD100K thành cases.json
├── tests/                    # 42 test chạy offline bằng client giả
├── images/                   # Ảnh thử nghiệm (test.jpg, test2.jpg, crop_test.jpg...)
├── data/
│   └── bdd100k/val/          # Dataset BDD100K validation (570MB ảnh + labels JSON)
├── reports/                  # Báo cáo dự án (thư mục mới)
└── results/                  # Output các lần chạy (overlays, crops, study_*.json/.md)
```

---

## 4. Công việc đã thực hiện

### 4.1 Bước A: Kiểm tra nhãn bằng ảnh overlay (HOÀN THÀNH)

Chạy `python tools/draw_boxes.py` tạo 2 file overlay:
- `results/overlays/test2_overlay.jpg` (c01–c05)
- `results/overlays/test_overlay.jpg` (c06–c07)

Kết quả đánh dấu `verified` trong `eval/cases.json`:

| Ca | Ảnh | Geometry | verified | Ghi chú |
|----|-----|----------|----------|---------|
| c01_taxi | test2.jpg | Trong khung, không chạm biên | ✅ true | |
| c02_car_behind_taxi | test2.jpg | Trong khung | ✅ true | |
| c03_suv_right_edge | test2.jpg | x2=2000 = chiều rộng → chạm biên phải → `truncated=true` khớp | ✅ true | |
| c04_motor_left | test2.jpg | x1=8, gần biên trái | ❌ false | Cần xem ảnh overlay xác nhận |
| c05_far_small_car | test2.jpg | Trong khung | ✅ true | |
| c06_rain_car_hidden | test.jpg | Trong khung | ✅ true | |
| c07_rain_motor | test.jpg | Trong khung | ✅ true | |

Luật hình học `touches_border` khớp 7/7 ca (100%, CI95 65–100%).

### 4.2 Bước B: Chạy thử nghiệm với 7 ca (HOÀN THÀNH)

Lệnh: `python eval/study.py --repeats 3 --tag pad30`

- 7 ca × 3 lần lặp = 21 lần gọi API
- Model: Qwen2.5-VL-7B-Instruct
- Độ trễ trung bình: 0.86s/lần

**Kết quả chính:**

| Chỉ số | Kết quả | CI95 |
|--------|---------|------|
| JSON parse | 100% | 84–100% |
| Đúng schema | 67% | 45–83% |
| Đúng lớp | 86% | 65–95% |
| Occlusion đúng | 60% | 36–80% |
| Báo che oan | 44% | 19–73% |
| Recall occlusion | 67% | 30–90% |
| Nhất quán 3 lần lặp | 71% | 36–92% |

File output: `results/study_pad30_20261118_112100.json` và `.md`

### 4.3 Bước C: Tải BDD100K và chạy 60 ca (HOÀN THÀNH)

1. Tải `huggingface_hub` và dataset `hirundo-io/bdd100k-val` (570MB)
2. Giải nén, tải `bdd100k_labels_images_val.json` (208MB)
3. Chạy `python tools/bdd_to_cases.py --n 60` → 60 cases (30 bị che, 30 không che)
4. Chạy `python eval/study.py --cases eval/cases_bdd.json --repeats 3 --tag bdd60`

- 60 ca × 3 lần lặp = 180 lần gọi API
- Độ trễ trung bình: 0.82s/lần

**Kết quả chính:**

| Chỉ số | Kết quả | CI95 |
|--------|---------|------|
| JSON parse | 100% | 98–100% |
| Đúng schema | 93% | 89–96% |
| Đúng lớp | 94% | 89–97% |
| Occlusion đúng | 57% | 49–64% |
| **Báo che oan** | **77%** | 67–84% |
| Recall occlusion | 90% | 82–95% |
| Đúng ánh sáng | 57% | 49–64% |
| Nhất quán 3 lần lặp | 92% | 82–96% |
| Luật hình học truncation | 97% | 89–99% |

File output: `results/study_bdd60_20261118_113345.json` và `.md`

---

## 5. Phân tích kết quả & kết luận

### 5.1 Điểm mạnh của Qwen2.5-VL-7B

| Tiêu chí | Kết quả | Đánh giá |
|----------|---------|----------|
| JSON parse | 100% | Xuất sắc, không lỗi định dạng |
| Đúng lớp xe | 94% | Rất tốt, dùng được cho phân loại đối tượng |
| Recall occlusion | 90% | Rất cao, ít bỏ sót trường hợp bị che |
| Nhất quán (3 lần lặp) | 92% | ổn định, ít biến động ngẫu nhiên |
| Luật hình học truncation | 97% | Mã xác định từ toạ độ bbox khớp GT gần hoàn hảo |

### 5.2 Điểm yếu cần lưu ý

| Tiêu chí | Kết quả | Đánh giá |
|----------|---------|----------|
| Báo che oan | 77% | **Rất cao** – VLM thường coi che giapan nhỏ (gương, biển số) là "occluded" |
| Đúng occlusion tổng thể | 57% | Chưa đủ chính xác làm nguồn sự thật |
| Đúng ánh sáng | 57% | Chưa đủ tin cậy phân loại bright/dim/dark |
| Đúng schema | 93% | 7% phản hồi vi phạm ràng buộc |

### 5.3 Khuyến nghị quyết định

**VLM KHÔNG NÊN** được dùng làm nguồn xác định occlusion độc lập trong Detection Doctor do tỷ lệ báo oan cao (77%).

**VLM CÓ THỂ hữu ích cho:**
- Phát hiện các trường hợp "có thể bị che" (recall cao 90%) để kỹ sư xem xét lại
- Cung cấp evidence/mô tả chi tiết về cách đối tượng bị che
- Phân loại loại xe (94% chính xác)
- Giải thích nguyên nhân lỗi kèm frame (sau cùng, bị chặn bởi luật vật lý/xác định)

**Những việc giao cho mã xác định (không giao cho VLM):**
- TP/FP/FN, IoU, AP/NDS
- Khoảng cách (metadata hoặc dữ liệu 3D)
- Bị cắt bởi biên ảnh (`touches_border` từ toạ độ bbox)

---

## 6. Kế hoạch tách 8 PR

Mỗi PR cần ít nhất 1 reviewer (không phải tác giả) để tính vào mốc **≥ 10 PR**.

| PR | Nội dung | File liên quan | Trạng thái |
|----|----------|----------------|------------|
| 1 | Client và parsing | `vlm/client.py`, `vlm/parsing.py` | Chưa tạo |
| 2 | Geometry | `vlm/geometry.py` | Chưa tạo |
| 3 | Scene_checks và main.py | `vlm/scene_checks.py`, `main.py` | Chưa tạo |
| 4 | Sửa evaluate.py | `eval/evaluate.py` | Chưa tạo |
| 5 | Probe và metrics | `vlm/probe.py`, `vlm/metrics.py` | Chưa tạo |
| 6 | Study và cases | `eval/study.py`, `eval/cases.json`, `eval/cases_bdd.json` | Chưa tạo |
| 7 | BDD to cases | `tools/bdd_to_cases.py` | Chưa tạo |
| 8 | README và cấu hình | `README.md`, `.env.example`, `requirements.txt` | Chưa tạo |

---

## 7. File output quan trọng

| File | Đường dẫn | Mô tả |
|------|-----------|-------|
| Kết quả 7 ca | `results/study_pad30_20261118_112100.json/.md` | Thí nghiệm pipeline ban đầu |
| Kết quả 60 ca BDD100K | `results/study_bdd60_20261118_113345.json/.md` | Thí nghiệm chính có ý nghĩa thống kê |
| Ảnh overlay kiểm tra | `results/overlays/test2_overlay.jpg`, `test_overlay.jpg` | Dùng để verify nhãn bằng mắt |
| Crop ảnh probe | `results/crops/dryrun3/` (dry-run), `results/crops/bdd60/` (live) | Ảnh crop gửi cho VLM |
| Cases BDD100K | `eval/cases_bdd.json` | 60 cases đã verified từ dataset thật |
| Báo cáo này | `reports/BAO_CAO_QWEN_VLM.md` | Báo cáo toàn diện dự án |

---

## 8. Giới hạn đã biết

1. Với n=60, khoảng tin cậy 95% vẫn tương đối rộng (ví dụ: báo oan 67–84%). Cần nhiều ca hơn nếu muốn kết luận chắc chắn.
2. Bộ 7 ca ban đầu là bản nháp ước lượng bằng mắt; chỉ 6/7 đã verified (c04 cần xem ảnh overlay).
3. `eval/sample_bdd_labels.json` là dữ liệu mẫu tổng hợp, không phải BDD100K thật.
4. Ghép trong `evaluate.py` chỉ là heuristic vì output cả cảnh không có bbox.
5. Chi phí và độ trễ mỗi lần gọi API cần được cân nhắc theo ngân sách của dự án.

---

**Kết luận tổng thể:** Thí nghiệm khả thi đã hoàn thành với số liệu đầy đủ từ 60 ca BDD100K. Qwen2.5-VL-7B có điểm mạnh về parse JSON, phân loại lớp và recall occlusion, nhưng tỷ lệ báo che oan cao (77%) làm nó không phù hợp làm nguồn sự thật độc lập cho occlusion. Nên tích hợp VLM như một thành phần hỗ trợ (gợi ý, giải thích), không phải quyết định cuối cùng.