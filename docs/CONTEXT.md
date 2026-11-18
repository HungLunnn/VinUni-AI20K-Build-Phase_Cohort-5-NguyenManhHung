> **Tài liệu bối cảnh gốc** (dùng làm prompt cho AI hỗ trợ). README thật nằm ở `qwen-vlm-poc/README.md`. Một số mục (cấu trúc thư mục, danh sách hạn chế) đã cũ so với code hiện tại.

# PROJECT CONTEXT – DETECTION DOCTOR (QWEN2.5-VL)

## 1. Tổng quan dự án

**Tên dự án:** Detection Doctor

**Lĩnh vực:** ADAS – Advanced Driver Assistance Systems / Computer Vision / Object Detection.

**Mục tiêu:** Xây dựng công cụ hỗ trợ kỹ sư phân tích và chẩn đoán các lỗi của mô hình Object Detection trong hệ thống hỗ trợ lái xe và xe tự hành.

Một mô hình Object Detection có mAP hoặc NDS cao vẫn có thể bỏ sót các đối tượng quan trọng, chẳng hạn:
- Người đi xe máy bị phương tiện khác che khuất.
- Phương tiện ở xa.
- Đối tượng có kích thước nhỏ.
- Đối tượng xuất hiện trong điều kiện ánh sáng hoặc thời tiết bất lợi.

Detection Doctor hướng tới việc giúp kỹ sư hiểu mô hình đang gặp vấn đề gì, vấn đề xuất hiện trong điều kiện nào và có bằng chứng hình ảnh nào hỗ trợ nhận định đó.

## 2. Phần việc tôi đang phụ trách

Tôi đang tập trung nghiên cứu và triển khai thành phần **VLM (Vision-Language Model), sử dụng Qwen2.5-VL**.

Phạm vi công việc hiện tại chỉ tập trung vào Qwen VLM, chưa triển khai các thành phần khác như embedding, clustering hoặc VISLIX.

### Vai trò của Qwen2.5-VL

Qwen được sử dụng để:
1. Phân tích hình ảnh giao thông.
2. Nhận diện và mô tả các phương tiện xuất hiện trong ảnh.
3. Mô tả vị trí tương đối của phương tiện.
4. Phát hiện dấu hiệu che khuất (occlusion).
5. Phân biệt đối tượng bị che khuất với đối tượng bị cắt bởi biên ảnh.
6. Mô tả các yếu tố môi trường có thể ảnh hưởng đến khả năng phát hiện đối tượng.
7. Trả về kết quả có cấu trúc JSON để các thành phần khác có thể sử dụng.

**Lưu ý kiến trúc quan trọng:**

Qwen không được phép tự quyết định TP, FP, FN hoặc IoU. Những kết quả này phải do Deterministic Evaluation Engine tính toán bằng Ground Truth và Prediction.

Qwen chỉ cung cấp mô tả hình ảnh, bằng chứng trực quan và các yếu tố có thể liên quan đến lỗi detection.

Qwen cũng không được tự suy đoán khoảng cách phương tiện nếu không có dữ liệu 3D hoặc metadata hỗ trợ.

## 3. Môi trường triển khai

Tôi lựa chọn chạy Qwen thông qua Cloud API thay vì chạy local vì cấu hình máy tính không phù hợp để chạy VLM lớn.

**Cấu hình máy tính:**
- CPU: Xeon E5-2696 v3.
- RAM: 32 GB.
- GPU: AMD RX570.

**Công nghệ sử dụng:**
- Python.
- OpenAI-compatible API.
- Requests.
- Python-dotenv.
- Pillow.
- FPT Cloud Marketplace.

**Model:**
`Qwen2.5-VL-7B-Instruct`

**API Base URL:**
`https://mkp-api.fptcloud.com/v1`

**Thư mục project:**
`D:\Vin\DetectionDoctor\qwen-vlm-poc`

Cấu trúc project dự kiến:

```text
qwen-vlm-poc/
│
├── .venv/
├── .env
├── .gitignore
├── main.py
├── check_env.py
│
├── images/
│   ├── test.jpg
│   └── crop_test.jpg
│
├── prompts/
│   └── traffic_analysis.txt
│
└── results/
    ├── first_response.txt
    └── crop_response.txt
```

File `.env` chứa API key:

```env
FPT_API_KEY=your_api_key_here
```

Không đưa API key vào source code hoặc GitHub.

## 4. Code hiện tại hoạt động như thế nào?

Luồng xử lý cơ bản:

```text
Input Image
    |
    v
Load Image
    |
    v
Convert Image to Base64
    |
    v
Load Prompt from TXT
    |
    v
Build API Request
    |
    v
Send Request to FPT Cloud API
    |
    v
Qwen2.5-VL-7B-Instruct
    |
    v
Receive Model Response
    |
    v
Parse JSON / Text Response
    |
    v
Save Result
```

### Giải thích các bước

**Bước 1 – Load ảnh**

Đọc ảnh giao thông từ thư mục `images/`.

**Bước 2 – Encode ảnh**

Chuyển ảnh thành Base64 và tạo Data URL:

`data:image/jpeg;base64,...`

**Bước 3 – Load prompt**

Đọc nội dung hướng dẫn từ:

`prompts/traffic_analysis.txt`

Tách prompt khỏi source code để dễ dàng chỉnh sửa và so sánh nhiều phiên bản prompt.

**Bước 4 – Gọi API**

Gửi prompt và ảnh tới endpoint `/chat/completions` bằng HTTP POST.

Request sử dụng:
- Model: Qwen2.5-VL-7B-Instruct.
- Temperature: 0.2.
- Max tokens: 1500.
- Stream: False.

**Bước 5 – Nhận kết quả**

Đọc nội dung từ response của API, lấy phần `choices[0].message.content` (có xử lý cấu trúc `data` nếu API trả về).

**Bước 6 – Lưu kết quả**

Lưu output để đánh giá và so sánh giữa các lần thử nghiệm.

## 5. Prompt hiện tại yêu cầu Qwen làm gì?

Prompt được thiết kế để Qwen phân tích hình ảnh giao thông và trả về JSON.

Các thông tin cần phân tích:

### A. Vehicles

Mỗi phương tiện cần có:
- `object_id`: ID tạm thời của đối tượng.
- `class`: Loại phương tiện.
- `position`: Vị trí tương đối.
- `visibility`: Trạng thái hiển thị.
- `occluder`: Đối tượng gây che khuất.
- `visible_parts`: Các bộ phận nhìn thấy.
- `hidden_parts`: Các bộ phận bị khuất.
- `evidence`: Bằng chứng hình ảnh.
- `uncertainty`: Thông tin chưa chắc chắn.

### B. Scene analysis

Phân tích:
- `traffic_density`: Mật độ giao thông.
- `lighting`: Điều kiện ánh sáng.
- `road_condition`: Tình trạng mặt đường.

### C. Possible detection difficulties

Các yếu tố có thể gây khó khăn cho Object Detection:
- Occlusion.
- Ánh sáng.
- Thời tiết.
- Mật độ giao thông.

Mỗi yếu tố phải có bằng chứng hình ảnh và mức độ confidence.

### D. Limitations

Qwen phải ghi rõ những điều không thể xác định chắc chắn từ hình ảnh.

Các trạng thái visibility đang được sử dụng:
- `fully_visible`
- `partially_occluded`
- `truncated_by_image_boundary`
- `uncertain`

## 6. Những thử nghiệm đã thực hiện

### Experiment 1 – Generic Traffic Analysis

**Mục tiêu:** Kiểm tra khả năng hiểu hình ảnh giao thông cơ bản của Qwen.

Kết quả:
- Nhận diện được các loại phương tiện.
- Có thể mô tả bối cảnh giao thông.
- Tuy nhiên, mô tả còn chung chung.
- Có khả năng mô tả trùng lặp đối tượng.
- Chưa phân tích tốt tình trạng che khuất.

### Experiment 2 – Structured JSON Prompt v1

**Mục tiêu:** Yêu cầu Qwen trả về kết quả có cấu trúc thay vì mô tả tự do.

Kết quả:
- Trả về JSON với các trường được yêu cầu.
- Nhận diện nhiều phương tiện.
- Có mô tả visible parts và hidden parts.
- Tuy nhiên, có một số đối tượng bị mô tả trùng lặp.
- Evidence chưa đủ cụ thể.
- Confidence chưa được kiểm chứng.

### Experiment 3 – Structured JSON Prompt v2

**Mục tiêu:** Cải thiện prompt bằng cách yêu cầu:
- Vị trí trái, giữa, phải.
- Hạn chế mô tả trùng lặp.
- Bổ sung bằng chứng cụ thể.
- Phân biệt quan sát và giả thuyết.
- Không tự suy đoán khoảng cách.
- Không tự xác định TP/FP/FN.

Kết quả:
- JSON có cấu trúc tốt hơn.
- Vị trí và evidence được mô tả rõ hơn.
- Số lượng phương tiện đầu ra thay đổi so với prompt v1.
- Vẫn có sự không nhất quán trong việc xác định occlusion.
- Một số trường hidden_parts và visibility chưa thống nhất về ý nghĩa.

### Experiment 4 – Full Image vs Crop Image

**Mục tiêu:** Kiểm tra liệu crop vùng phương tiện có giúp Qwen phân tích chi tiết hơn ảnh toàn cảnh hay không.

Sử dụng cùng prompt và cấu hình API để hạn chế các yếu tố gây nhiễu.

Kết quả thử nghiệm ảnh crop gần nhất:
- Qwen trả về 4 phương tiện: 1 car và 3 motorcycles.
- Bỏ sót nhiều xe máy xuất hiện trong ảnh.
- Nhận định ô tô ở giữa ảnh, trong khi thực tế ô tô nằm bên phải và bị cắt một phần bởi biên ảnh.
- Đánh dấu các phương tiện là fully_visible chưa chính xác.
- Kết luận không có occlusion đáng kể dù ảnh có nhiều phương tiện đứng sát và che khuất nhau.

**Nhận xét:** Crop ảnh chưa đảm bảo cải thiện chất lượng phân tích. Qwen vẫn gặp khó khăn trong việc đếm phương tiện, phân biệt từng đối tượng và xác định trạng thái hiển thị.

## 7. Các hạn chế hiện tại

1. Qwen có thể bỏ sót phương tiện trong cảnh giao thông đông đúc.
2. Có thể nhầm lẫn giữa các phương tiện đứng gần nhau.
3. Chưa xác định occlusion một cách ổn định.
4. Có thể đánh dấu sai đối tượng bị cắt bởi biên ảnh.
5. Evidence đôi khi còn chung chung, chưa chỉ ra bộ phận hoặc vị trí cụ thể.
6. Confidence do mô hình tự đưa ra chưa được calibration.
7. Chưa có Ground Truth để xác nhận độ chính xác của kết quả.

Cần phân biệt:
- JSON hợp lệ về mặt cú pháp.
- Output tuân thủ prompt.
- Nội dung mô tả đúng với hình ảnh.
- Kết quả chính xác khi đối chiếu Ground Truth.

Bốn tiêu chí trên không đồng nghĩa với nhau.

## 8. Hướng phát triển tiếp theo

Tôi muốn tiếp tục hoàn thiện riêng thành phần Qwen VLM theo thứ tự:

1. Cải thiện prompt thành phiên bản v3.
2. Bắt buộc mô hình phân biệt rõ fully visible, partially occluded và truncated by image boundary.
3. Yêu cầu evidence cụ thể hơn, tránh những câu chung chung như "clearly visible".
4. Thử nghiệm trên ảnh toàn cảnh và ảnh crop với cùng cấu hình.
5. Thử nghiệm thêm các ảnh có occlusion rõ ràng, phương tiện nhỏ hoặc bị cắt một phần.
6. Lưu kết quả từng lần chạy thành JSON để thuận tiện so sánh.
7. Xây dựng bảng đánh giá chất lượng output.
8. Viết tài liệu mô tả phương pháp, kết quả và hạn chế của Qwen VLM PoC.

## 9. Yêu cầu đối với AI hỗ trợ tiếp tục dự án

Bạn hãy đóng vai trò AI hỗ trợ nghiên cứu và phát triển thành phần Qwen VLM của Detection Doctor.

Các nguyên tắc làm việc:

- Đọc và hiểu code hiện tại trước khi đề xuất thay đổi.
- Không tự giả định code đã có những chức năng chưa được triển khai.
- Khi sửa code, phải chỉ rõ file nào, vị trí nào và nội dung cần thay thế.
- Giải thích mục đích của từng thay đổi.
- Hướng dẫn từng bước, không đưa quá nhiều công việc cùng lúc.
- Sau mỗi bước, chờ tôi chạy thử và gửi kết quả rồi mới tiếp tục.
- Không yêu cầu tôi chia sẻ API key.
- Không khẳng định Qwen nhận diện chính xác nếu chưa có Ground Truth.
- Không để Qwen tự quyết định TP, FP, FN, IoU hoặc khoảng cách phương tiện.
- Ưu tiên các thử nghiệm có thể đo lường và so sánh được thay vì chỉ đánh giá cảm tính.

**Mục tiêu cuối cùng:** Hoàn thiện một Qwen2.5-VL Proof of Concept có khả năng phân tích hình ảnh giao thông, trả về dữ liệu JSON có cấu trúc, cung cấp evidence trực quan và hỗ trợ kỹ sư tìm hiểu nguyên nhân có thể dẫn đến lỗi Object Detection trong hệ thống ADAS.