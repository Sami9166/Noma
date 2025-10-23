<img width="1920" height="424" alt="제목을 입력해주세요  (4)" src="https://github.com/user-attachments/assets/1b5fc56b-c5ec-486c-aef7-6bf25b1e40aa" />

# NoMa
데이터로 완성하는 당신만의 AI 마케팅 비서

## Website Link
https://2025-bigcon-noma.streamlit.app/
- streamlit cloud로 배포되었습니다.

## Local Start
### Environment Variables
- You need to set your own API key

```bash
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
```
### Setting
  - [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)로 설치하면 됩니다.

```bash
# Linux and macOS
git clone https://github.com/Sami9166/2025_bigcon
cd 2025_bigcon

uv venv
source .venv/bin/activate

uv pip install -r requirements.txt

uv run streamlit run app.py
```
```bash
# Window
git clone https://github.com/Sami9166/2025_bigcon
cd 2025_bigcon

uv venv
call .venv/bin/activate

uv pip install -r requirements.txt

uv run streamlit run app.py
```
