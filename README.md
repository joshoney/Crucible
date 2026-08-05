# DistVal: Distributed Agentic Model Evaluation Platform

## What It Is
DistVal is a distributed orchestration platform designed to automate the evaluation of Large Language Models (LLMs) and publish the results.

At its core, DistVal leverages the **Model Context Protocol (MCP)** to allow external agents to trigger evaluations. These requests are picked up by a **Temporal** orchestration engine which manages a complex, multi-step workflow. The system provisions models on local hardware (via a Lemonade server), executes benchmarking using `evaluerBench`, offloads large evaluation artifacts to an S3-compatible **MinIO** instance, and finally publishes the results atomically directly to a GitHub repository (such as an Astro site) for public viewing.

---

## How to Run It

### Prerequisites
Before running DistVal, ensure your environment meets the following requirements:
- **Docker & Docker Compose**: The entire infrastructure runs in containers.
- **Python 3.10+** (if running the Python scripts locally instead of in Docker).

### Configuration
DistVal relies on environment variables to connect to external services. Create or update the `.env` file in the root directory:

```env
# Tracing for evaluerBench
LANGCHAIN_API_KEY=your_langchain_api_key_here

# Target GitHub Repository for publishing results
GITHUB_TOKEN=your_github_token_here
GITHUB_REPO=your_username/your-astro-repo
TARGET_BRANCH=main # Optional, defaults to main

# Local Hardware Server for model provisioning
LEMONADE_API_URL=http://10.10.0.12:8000/api/v1
```

### Starting the Infrastructure
To launch the entire platform, run the following command from the root directory:

```bash
docker-compose up -d --build
```

This will spin up all the necessary components:
1. **PostgreSQL** (State store for Temporal)
2. **Temporal Server** (Orchestration Engine on port `7233`)
3. **Temporal UI** (Dashboard available at `http://localhost:8080`)
4. **MinIO** (S3 Storage API on `9000`, Web UI at `http://localhost:9001`)
5. **MinIO Init** (Ephemeral container that auto-creates the `eval-artifacts` bucket)
6. **eval-worker** (The Temporal worker executing the evaluation logic)
7. **mcp-gateway** (The FastMCP server exposing tools on port `8081`)

To stop the infrastructure, run:
```bash
docker-compose down
```

---

## Architecture & Workflow Deep Dive

DistVal separates concerns to ensure scalability, fault tolerance, and deterministic execution.

### Core Architecture Components
1. **MCP Gateway (`mcp-gateway`)**: A lightweight FastMCP Python server. It exposes tools (`test_model` and `get_evaluation_status`) to AI agents. It does *not* execute logic itself; it simply pushes a workflow request into the Temporal Task Queue and returns a tracking `taskID`.
2. **Temporal Control Plane**: Acts as the brain of the operation. It tracks the state, handles retries, schedules timeouts, and queues tasks for workers.
3. **Eval Orchestrator (`eval-worker`)**: The heavy lifter. A Python worker that listens to the Temporal queue and executes the defined activities. Concurrency is strictly limited to protect local hardware resources.
4. **MinIO**: Prevents the Temporal database from becoming bloated. Large evaluation artifacts (JSON traces, markdown files, logs) are pushed to MinIO instead of being passed as workflow state.

### The `LlmEvaluationWorkflow` Process
When a request hits the MCP gateway, the `LlmEvaluationWorkflow` is triggered. It guarantees completion through the following 3-step pipeline:

#### 1. Provision Model
The worker makes a request to the local `LEMONADE_API_URL` to provision the requested model (e.g., `llama-3-8b-instruct`) onto the local LAN hardware. If the hardware server is busy or unreachable, Temporal's standard retry policy automatically backs off and retries.

#### 2. Run Agentic Evaluation
The worker triggers the `evaluerBench` suite against the provisioned model. 
- A localized scratchpad directory is created for the specific `taskID`.
- `evaluerBench` writes its extensive artifacts to this scratchpad.
- To maintain a lean Temporal state history, the worker walks the scratchpad and streams all artifacts directly into the local **MinIO** `eval-artifacts` bucket. 

#### 3. Publish Results
The worker completes the loop by publishing the artifacts to the public internet:
- It pulls the artifacts belonging to the `taskID` from MinIO.
- It maps `.json` files to `results/{taskID}/` and all other artifacts to `public/artifacts/{taskID}/`.
- Using the GitHub REST API (Git Data API), it constructs an atomic tree and pushes a single, clean commit to the configured `GITHUB_REPO`. 

Because Temporal tracks every step, if the GitHub API fails due to rate-limiting, Temporal will pause the workflow and retry the `publish_results` activity without needing to re-evaluate the model.
