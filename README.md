# Agent CSV Analyzer

Đồ án môn Chuyên đề các vấn đề hiện đại trong CNTT.
Web app dùng AI Agent (smolagents CodeAgent) phân tích file CSV theo mục tiêu nhập bằng tiếng Việt.

## Chức năng

- Upload file CSV, xem số dòng, số cột và bảng dữ liệu
- Nhập mục tiêu phân tích bằng tiếng Việt
- Agent tự chọn và gọi 4 tool phân tích:
  - `profile_dataframe`: kích thước, kiểu dữ liệu, giá trị thiếu
  - `describe_numeric_columns`: thống kê mô tả các cột số
  - `compute_correlation`: tương quan Pearson (không phải quan hệ nhân quả)
  - `plot_histogram`: vẽ histogram một cột số (Plotly)
- Số liệu trong báo cáo lấy từ kết quả tool, không do LLM tự tạo
- Mỗi lần chạy được ghi vào `run_log.txt` (thời gian, số bước, trạng thái)

## Mô hình hỗ trợ

- `gemini-flash-lite-latest` (Gemini API)
- `groq-qwen3.8-27b` (Groq API)

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc:

```text
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

> Không commit file `.env` (đã có trong `.gitignore`).

## Chạy

```bash
streamlit run app.py
```

## Hạn chế

- Chỉ có biểu đồ histogram
- Không tự làm sạch dữ liệu, chỉ phát hiện và cảnh báo
- Dữ liệu thống kê được gửi tới nhà cung cấp LLM; không dùng với dữ liệu nhạy cảm