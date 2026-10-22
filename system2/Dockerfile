# System2 (Qwen-VL planner) inference image: vLLM 0.19.1 only.
#
# Same public ModelScope base as sys2_train_eval/docker/eval.Dockerfile, which validated System2
# generation with this exact vLLM build; the SWIFT training/PPL venv that image adds is not needed
# to serve. Every serve-time media setting is passed by scripts/serve_system2.sh, so the image itself
# carries no model or repo code.
#
#   docker build -t sys1-sys2/system2-vllm:0.19.1 system2
ARG BASE_IMAGE=modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.9.1-py312-torch2.10.0-vllm0.19.1-modelscope1.35.4-swift4.1.3
FROM ${BASE_IMAGE}

RUN /usr/local/bin/python -c 'import importlib.metadata as m; v = m.version("vllm"); assert v == "0.19.1", v; print("vllm", v)'

ENTRYPOINT ["/usr/local/bin/vllm"]
