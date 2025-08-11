FROM stablebaselines/rl-baselines3-zoo:latest

# Install sai-rl
# Original container uses micromamba and uv
RUN \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    uv pip install --system sai-rl && \
    uv pip install --system sai-mujoco && \
    uv pip install --system mlflow && \
    uv pip install --system optuna && \
    uv pip install --system optuna-dashboard && \
    uv cache clean

WORKDIR /workspace

CMD ["/bin/bash"]