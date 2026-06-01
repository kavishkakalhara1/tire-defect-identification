FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir streamlit pillow

COPY main.py /app/main.py
COPY streamlit_app.py /app/streamlit_app.py
COPY tire_defect_resnet18.pth /app/tire_defect_resnet18.pth
COPY tire_defect_resnet18_scripted.pt /app/tire_defect_resnet18_scripted.pt
COPY tire_defect_resnet18_traced.pt /app/tire_defect_resnet18_traced.pt
COPY README.md /app/README.md

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]