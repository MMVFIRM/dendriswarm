FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
RUN pip install --no-cache-dir ".[coordinator]"
EXPOSE 8787
ENTRYPOINT ["dendriswarm"]
