FROM registry.access.redhat.com/ubi9/python-39:latest

USER 0

WORKDIR /app
COPY requirements.txt /app

RUN chown -R 1001:0 ./
USER 1001

RUN pip install -r requirements.txt
RUN mkdir SOURCE_DOCUMENTS
RUN mkdir DB
RUN mkdir DB.quarkus

COPY DB DB
COPY DB.quarkus DB.quarkus

COPY run_GPTUI.py .
COPY .chainlit .
COPY chainlit.md .
COPY constants.py .

EXPOSE 8000
ENV OPENAI_API_KEY set_your_own

ENTRYPOINT ["chainlit", "run", "/app/run_GPTUI.py"]