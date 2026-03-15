# 🎙️ Voice Escalation Copilot (Powered by Amazon Nova)

> **Amazon Nova Hackathon Submission**
> Transforming complex support interactions into structured, instantly actionable escalation tickets using **Amazon Nova Sonic** and **Nova Lite**.

## 🌟 The Problem
Customer support escalations are messy. Disjointed facts, missing context, and lengthy back-and-forth between Tier 1 agents and specialized teams result in SLA breaches and frustrated customers. When agents type out notes manually while talking, critical context is lost.

## 🚀 Our Solution
**Voice Escalation Copilot** acts as a structured "second pair of ears". Instead of letting the agent freestyle ticket creation, the Copilot (powered by **Nova Sonic**) conducts a rapid, structured 6-question intake interview.
1. **Intake (Nova Sonic)**: Bidirectional, speech-to-speech interaction asks specific questions (Timeline, Customer Impact, Urgency) and provides contextual acknowledgements.
2. **Analysis (Nova Lite)**: Consolidates the intake interview into a structured brief. Nova Lite analyzes sentiment, extracts metadata, flags SLA/Churn risks, and determines the optimal routing destination.
3. **Escalation Queue**: An actionable dashboard where supervisors can instantly review, approve, and act on AI-generated escalation drafts.

---

## 🏗️ Architecture

- **Backend (`/backend`)**: Built with **FastAPI**. Interfaces with the AWS Bedrock Runtime to stream audio via the `BidirectionalStream` API (for Nova Sonic) and calls Nova Lite for structured JSON output.
- **Frontend (`/frontend`)**: A modern **React** (Vite) application with a beautiful, premium glassmorphic UI, responsive state management, and an integrated Web Speech API for voice dictation.
- **Local Storage / DB**: Stores ticket data locally in a JSON file for blazing-fast demonstration purposes (`backend/data/tickets.json`).

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- AWS Account with Bedrock access to:
  - `amazon.nova-2-sonic-v1:0`
  - `amazon.nova-lite-v1:0`

### 1. Environment Setup
Create a `.env` file in the root of the project with your AWS credentials:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
# Or if using temporary credentials:
AWS_SESSION_TOKEN=your_session_token
```

### 2. Run the Backend
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
python -m uvicorn backend.app.api:app --host 0.0.0.0 --port 8000 --reload
```
The backend API documentation will be available at `http://localhost:8000/docs`.

### 3. Run the Frontend
```powershell
cd frontend

# Install packages
npm install

# Start the Vite dev server
npx vite --port 4300
```
Open `http://localhost:4300` in your browser.

---

## 💡 Key Features of the App
- **Real-time Sonic Processing**: Uses Bedrock's bidirectional streaming to simulate a live AI copilot interview.
- **Dynamic Routing & Risk Detection**: AI automatically calculates the correct escalation path, severity level, and flags potential churn.
- **Fallback Resilience**: Gracefully handles API downtime or missing SDKs with built-in text-fallback mechanisms.
- **Voice Dictation**: Uses the browser's native speech recognition so agents can speak instead of type.

---

## 🏆 Hackathon Notes
We built this project to highlight the specific low-latency, conversational capabilities of **Nova Sonic**, combined with the strict JSON formatting capabilities of **Nova Lite**. The UI is built to look "production-ready" to demonstrate real business utility.
