# Qwen2.5-VL: nghiên cứu khả thi cho Detection Doctor

Thư mục này trả lời một câu hỏi: **có nên dùng Qwen2.5-VL-7B (FPT Cloud) làm thành phần giải thích lỗi của Detection Doctor không, và nên giao cho nó việc gì.** Đây là nghiên cứu, không nằm trên đường chính của MVP 22/11. Mục tiêu là để khi tích hợp VLM sau này, nhóm đã có số đo thay vì cảm tính.

## Kết luận hiện tại (từ các thí nghiệm đã chạy; cần kiểm lại khi có số liệu mới)

| Phát hiện | Bằng chứng | Hệ quả |
| --- | --- | --- |
| Phân tích cả cảnh đông xe không đáng tin | `results/response_v3_2.txt`: 7/7 xe `fully_visible`, 6 xe máy cùng visible/hidden parts, evidence chỉ khác vị trí; `vlm/scene_checks.py` bắt 14 mâu thuẫn nội tại và 1 nhóm lặp | Dừng tối ưu prompt cả cảnh |
| Cách đánh giá cũ ghép theo `object_id` | ID do Qwen tự đặt nên GT "V3" (ô tô) bị ghép với Qwen "V3" (xe máy) | `eval/evaluate.py` đã sửa: ghép theo lớp và vị trí; GT chưa đầy đủ thì không đếm FP |
| Thí nghiệm "crop vs ảnh gốc" cũ bị nhiễu | `crop_test.jpg` là góc (0,0,1000,1000) của `test2.jpg`, cắt đôi chiếc taxi; ở ảnh gốc taxi nằm trọn khung hình | Không kết luận về "truncated" từ ca đó |
| "Bị cắt bởi biên ảnh" không cần VLM | Suy ra từ toạ độ bbox: `vlm/geometry.py::touches_border` | Giao cho code, VLM chỉ trả lời về che khuất |

Đã có số liệu từ 60 ca BDD100K val (xem `reports/BAO_CAO_QWEN_VLM.md` và `results/study_bdd60_*.json/.md`). Kết quả chính: JSON parse 100%, đúng lớp 94%, recall occlusion 90%, nhưng báo che oan cao 77%.

## Phân công dự kiến: việc nào cho VLM, việc nào cho code

| Việc | Ai làm | Ghi chú |
| --- | --- | --- |
| TP/FP/FN, IoU, AP/NDS | Engine đánh giá xác định | VLM không được quyết định |
| Khoảng cách | Metadata hoặc dữ liệu 3D | VLM không ước lượng |
| Bị cắt bởi biên ảnh | Code (`touches_border`) | Đo độ khớp với nhãn BDD `truncated` bằng study |
| Có bị che không, che bởi gì | VLM, **cần đo** | Chỉ số: recall, báo oan, nhất quán |
| Ánh sáng, thời tiết | Metadata dataset hoặc VLM | So sánh hai nguồn |
| Giải thích nguyên nhân kèm frame | VLM, sau cùng | Bị chặn bởi luật hình học/vật lý xác định |

## Cách chạy

### Bước 1: Cài môi trường

```powershell
# Tạo và kích hoạt venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt

# Tạo file .env từ mẫu và điền FPT_API_KEY
cp .env.example .env
# Mở file .env và điền giá trị FPT_API_KEY của bạn
```

### Bước 2: Chạy test offline (không cần API)

```powershell
python -m unittest discover -v        # 42 test, chạy hoàn toàn offline
```

### Bước 3: Kiểm tra nhãn bằng ảnh overlay (bắt buộc trước khi chạy thật)

```powershell
python tools/draw_boxes.py            # vẽ bbox lên ảnh, lưu tại results/overlays/
# Mở file ảnh overlay để xác nhận bbox và nhãn, sau đó đặt verified: true trong eval/cases.json
```

### Bước 4: Chạy thí nghiệm probe

**Chỉ lưu crop, không gọi API (dry-run):**
```powershell
python eval/study.py --dry-run --repeats 3 --tag dryrun3
```

**Chạy thật với 7 ca ban đầu, 3 lần lặp/ca:**
```powershell
python eval/study.py --repeats 3 --tag pad30
```

**Chạy thật với 60 ca BDD100K (cần tải dataset trước, xem mục dưới):**
```powershell
python eval/study.py --cases eval/cases_bdd.json --repeats 3 --tag bdd60
```

**Chơi lại từ file kết quả cũ (không gọi API):**
```powershell
python eval/study.py --replay results/study_pad30_<timestamp>.json
```

### Bước 5: Tải BDD100K validation và tạo 60 cases

```powershell
# Cài huggingface_hub
pip install huggingface_hub

# Tải dataset BDD100K val (570MB, có thể mất vài phút)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hirundo-io/bdd100k-val', repo_type='dataset', local_dir='data/bdd100k/val')"

# Giải nén file zip
Expand-Archive -LiteralPath "data\bdd100k\val\bdd100k_val_hirundo.zip" -DestinationPath "data\bdd100k\val" -Force

# Tải file labels JSON chính thức
Invoke-WebRequest -Uri 'https://www.joligen.com/datasets/bdd100k/bdd100k_labels_images_val.json.zip' -OutFile 'data\bdd100k\val\bdd100k_labels_images_val.json.zip'
Expand-Archive -LiteralPath "data\bdd100k\val\bdd100k_labels_images_val.json.zip" -DestinationPath "data\bdd100k\val" -Force

# Tạo 60 cases từ BDD100K (30 bị che, 30 không che)
python tools/bdd_to_cases.py --labels data\bdd100k\val\bdd100k_labels_images_val.json --images-dir ..\data\bdd100k\val\bdd100k\images\100k\val --n 60 --out eval\cases_bdd.json

# Chạy study với 60 ca
python eval/study.py --cases eval/cases_bdd.json --repeats 3 --tag bdd60
```

### Biến môi trường

| Biến | Bắt buộc/Tùy chọn | Mô tả |
|------|-------------------|-------|
| `FPT_API_KEY` | Bắt buộc | API key cho FPT Cloud Qwen2.5-VL |
| `FPT_BASE_URL` | Tùy chọn | Mặc định: `https://mkp-api.fptcloud.com/v1` |
| `FPT_MODEL` | Tùy chọn | Mặc định: `Qwen2.5-VL-7B-Instruct` |

Mỗi lần chạy sinh `results/study_<tag>_<thời gian>.json` (đủ để replay, gồm raw text từng lần gọi) và `.md` (bảng kết quả).

## Thêm ca thử nghiệm

`eval/cases.json`: mỗi ca là một đối tượng.

```json
{"id": "c01", "image": "images/test2.jpg", "bbox": [x1, y1, x2, y2],
 "gt": {"class": "car", "occluded": false, "truncated": false, "lighting": "bright"},
 "tags": ["car"], "verified": false}
```

Trường nào không chắc thì **bỏ khỏi `gt`**, đừng đoán: trường vắng mặt không bị chấm. Bộ 7 ca hiện có là **bản nháp** (bbox và nhãn ước lượng bằng mắt): kiểm tra bằng `tools/draw_boxes.py`, sửa nếu sai, rồi đặt `verified: true`.

Để có số ca đủ lớn, lấy nhãn thật từ BDD100K:

```bash
python tools/bdd_to_cases.py --labels bdd100k_labels_images_val.json --images-dir data/bdd100k/val --n 60
python eval/study.py --cases eval/cases_bdd.json --repeats 3 --tag bdd60
```

## Cách đọc kết quả

- Với n nhỏ, khoảng tin cậy 95% rất rộng (ví dụ 3/5 đúng cho CI khoảng 23%–88%). **7 ca chỉ để kiểm tra pipeline, không đủ kết luận.** Cần vài chục ca trở lên.
- Luôn so với mốc đơn giản: VLM luôn trả lời "không che" đạt bao nhiêu trên cùng tập?
- Bốn thứ khác nhau, không được gộp: JSON parse được, đúng schema, nhất quán nội bộ, đúng so với nhãn thật.
- Báo oan (nói che nhưng thực tế không) và bỏ sót che khuất có hậu quả khác nhau cho kỹ sư; xem riêng.

### Tiêu chí quyết định (đề xuất, nhóm cần thống nhất trước khi chạy)

| Chỉ số | Gợi ý ngưỡng để cân nhắc tích hợp |
| --- | --- |
| Parse và schema | ≥ 95% / ≥ 90% |
| Nhất quán giữa 3 lần lặp (temperature 0.2) | ≥ 80% |
| Occlusion: tốt hơn rõ rệt mốc "luôn không che" | cận dưới CI95 vượt mốc |
| Báo oan | ≤ 20% |
| Độ trễ và chi phí mỗi lần gọi | nhóm tự đặt theo ngân sách |

Các ngưỡng là gợi ý khởi đầu, không phải chuẩn công nghiệp.

## Cấu trúc

```text
vlm/            client.py (gọi API, retry), parsing.py, geometry.py, probe.py, metrics.py, scene_checks.py, labels.py
eval/           study.py (chạy nghiên cứu), cases.json (ca nháp), evaluate.py (so cảnh với GT), bdd100k_to_gt.py (cũ)
prompts/        object_probe_v1.txt (hướng mới), traffic_analysis*.txt (phân tích cả cảnh, giữ để so sánh)
tools/          draw_boxes.py, bdd_to_cases.py
tests/          42 test chạy offline bằng client giả
results/        output các lần chạy
```

## Tích hợp với Detection Doctor về sau

Hàm vào là `vlm.probe.probe_object(client, image, bbox, prompt)` trả về dict gồm `parsed` (schema ở `prompts/object_probe_v1.txt`), `schema_ok`, `schema_errors`, `raw_text`, `latency_s`. Engine đánh giá xác định cung cấp bbox của lỗi (ví dụ GT bị bỏ sót); VLM giải thích đối tượng đó; mọi giả thuyết phải qua tầng luật vật lý và kỹ sư duyệt, đúng ràng buộc của đề.

## Giới hạn đã biết

- Đã có kết quả chạy thật từ 60 ca BDD100K, nhưng n=60 vẫn để khoảng tin cậy 95% tương đối rộng (ví dụ báo oan 67–84%). Cần nhiều ca hơn nếu muốn kết luận chắc chắn.
- Bộ 7 ca ban đầu (`eval/cases.json`) là bản nháp ước lượng bằng mắt; chỉ 6/7 đã verified (c04 cần xem ảnh overlay).
- `eval/sample_bdd_labels.json` là dữ liệu mẫu tổng hợp, không phải BDD100K thật.
- Ghép trong `evaluate.py` chỉ là heuristic vì output cả cảnh không có bbox.
- Chi phí và độ trễ mỗi lần gọi API (~0.82s/lần) cần được cân nhắc theo ngân sách của dự án.
