# Voice Escalation Copilot

Hackathon project built for the Amazon Nova challenge.

Customer support escalations lose context fast. Agents typing notes while on a call miss critical details. This tool acts as a structured second pair of ears during the call.

**How it works**

Nova Sonic conducts a rapid 6-question intake interview with the agent in real time. Nova Lite then consolidates the conversation into a structured escalation brief, flags SLA and churn risks, and routes the ticket to the right team.

**Stack**

FastAPI · React · Amazon Nova Sonic · Amazon Nova Lite · AWS Bedrock

**Features**

- Bidirectional speech-to-speech intake via Nova Sonic
- Structured JSON escalation briefs from Nova Lite
- Auto-routing based on severity and churn risk
- Voice dictation via browser Web Speech API
- Fallback to text if API is unavailable
