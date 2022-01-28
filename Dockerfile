FROM python:3.7-slim

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -rrequirements.txt

ARG phase
ARG only_validate
ENV only_validate=$only_validate

COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator test/$phase/reference.csv /opt/evaluation/test/reference.csv

# ENTRYPOINT "python" "-m" "evaluation" "$only_validate"
ENTRYPOINT python evaluation.py $only_validate
