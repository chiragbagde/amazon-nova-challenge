import { useEffect, useMemo, useRef, useState } from "react";

type TicketCategory = "billing" | "delivery" | "technical" | "account" | "refund" | "other";
type TicketPriority = "low" | "medium" | "high" | "critical";
type EscalationLevel = "none" | "supervisor" | "specialist" | "incident";
type RiskLevel = "low" | "medium" | "high";
type CallPhase = "idle" | "questions" | "recording" | "processing" | "done";

interface SonicQuestion {
  key: string;
  label: string;
  spoken: string;
}

interface SonicAnswerItem {
  key: string;
  label: string;
  answer: string;
  nova_ack: string;
}

interface SonicIntakeResponse {
  answers: SonicAnswerItem[];
  transcript: string;
  full_text: string;
  success: boolean;
  errors: string[];
}

interface TicketDraft {
  title: string;
  summary: string;
  category: TicketCategory;
  priority: TicketPriority;
  sentiment: string;
  route_to: string;
  escalation_level: EscalationLevel;
  sla_risk: RiskLevel;
  churn_risk: RiskLevel;
  requires_manager_review: boolean;
  refund_requested: boolean;
  customer_impact: string;
  resolution_path: string;
  next_actions: string[];
  suggested_reply: string;
}

interface AnalyzeResponse {
  draft: TicketDraft;
  confidence: number;
  source: "nova" | "fallback";
  warnings: string[];
}

interface TicketRecord {
  id: string;
  created_at: string;
  transcript: string;
  draft: TicketDraft;
}

interface TicketListResponse {
  items: TicketRecord[];
}

type SpeechCtor = new () => {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: { results: { isFinal: boolean; 0: { transcript: string } }[] }) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const SAMPLE_ANSWERS = [
  "The customer says they were charged twice for the same safety harness order.",
  "The affected purchase is yesterday's order under rahul.verma@example.com, order #AZ-29841.",
  "The order was placed yesterday evening and both card charges are already completed.",
  "A large amount is blocked on the card and the customer is worried about cash flow.",
  "The customer wants this fixed today and may cancel their account if the refund is delayed.",
  "The customer wants the duplicate charge reversed immediately and confirmation of the refund timeline.",
];

// ── Waveform animation component ──────────────────────────────────────────────

function Waveform({ active }: { active: boolean }) {
  return (
    <div className={`waveform ${active ? "waveform--live" : ""}`} aria-hidden>
      {Array.from({ length: 12 }).map((_, i) => (
        <span key={i} className="waveform__bar" style={{ animationDelay: `${i * 0.07}s` }} />
      ))}
    </div>
  );
}

// ── Typing indicator ──────────────────────────────────────────────────────────

function NovaThinking() {
  return (
    <div className="nova-thinking" aria-live="polite">
      <span className="nova-dot" style={{ animationDelay: "0s" }} />
      <span className="nova-dot" style={{ animationDelay: "0.18s" }} />
      <span className="nova-dot" style={{ animationDelay: "0.36s" }} />
      <span className="nova-label">Nova is thinking…</span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

function App() {
  const [language, setLanguage] = useState("en");

  // Sonic interview state
  const [sonicQuestions, setSonicQuestions] = useState<SonicQuestion[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [agentAnswers, setAgentAnswers] = useState<string[]>(Array(6).fill(""));
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [callPhase, setCallPhase] = useState<CallPhase>("idle");
  const [sonicResult, setSonicResult] = useState<SonicIntakeResponse | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [compiledTranscript, setCompiledTranscript] = useState("");

  // Analysis state
  const [draft, setDraft] = useState<TicketDraft | null>(null);
  const [analysisMeta, setAnalysisMeta] = useState<Omit<AnalyzeResponse, "draft"> | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  // Queue state
  const [tickets, setTickets] = useState<TicketRecord[]>([]);
  const [isLoadingTickets, setIsLoadingTickets] = useState(false);

  const [errorText, setErrorText] = useState("");
  const [successText, setSuccessText] = useState("");

  const recognitionRef = useRef<InstanceType<SpeechCtor> | null>(null);

  const speechCtor = useMemo<SpeechCtor | null>(() => {
    const w = window as Window & {
      SpeechRecognition?: SpeechCtor;
      webkitSpeechRecognition?: SpeechCtor;
    };
    return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
  }, []);

  useEffect(() => {
    void Promise.all([fetchQuestions(), fetchTickets()]);
    return () => {
      if (recognitionRef.current) recognitionRef.current.stop();
    };
  }, []);

  const queueMetrics = useMemo(() => {
    const openCount = tickets.length;
    const managerReview = tickets.filter((t) => t.draft.requires_manager_review).length;
    const criticalCount = tickets.filter((t) => t.draft.priority === "critical").length;
    const slaHot = tickets.filter((t) => t.draft.sla_risk === "high").length;
    return { openCount, managerReview, criticalCount, slaHot };
  }, [tickets]);

  // ── Data fetching ───────────────────────────────────────────────────────────

  async function fetchQuestions(): Promise<void> {
    try {
      const res = await fetch(`${API_BASE_URL}/api/sonic/questions`);
      if (!res.ok) return;
      const data = (await res.json()) as { questions: SonicQuestion[] };
      setSonicQuestions(data.questions);
    } catch {
      /* non-fatal */
    }
  }

  async function fetchTickets(): Promise<void> {
    setIsLoadingTickets(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/tickets?limit=10`);
      if (!res.ok) throw new Error(`Could not load tickets (${res.status})`);
      const data = (await res.json()) as TicketListResponse;
      setTickets(data.items ?? []);
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : "Could not load tickets");
    } finally {
      setIsLoadingTickets(false);
    }
  }

  // ── Sonic interview flow ────────────────────────────────────────────────────

  function startCall(): void {
    setErrorText("");
    setSuccessText("");
    setCurrentQuestionIndex(0);
    setAgentAnswers(Array(6).fill(""));
    setCurrentAnswer("");
    setCallPhase("questions");
    setDraft(null);
    setAnalysisMeta(null);
    setSonicResult(null);
    setCompiledTranscript("");
  }

  function loadSampleAnswers(): void {
    setErrorText("");
    setSuccessText("");
    setCurrentQuestionIndex(0);
    setAgentAnswers([...SAMPLE_ANSWERS]);
    setCurrentAnswer(SAMPLE_ANSWERS[0]);
    setCallPhase("questions");
    setDraft(null);
    setAnalysisMeta(null);
    setSonicResult(null);
    setCompiledTranscript("");
  }

  function saveAnswer(): void {
    if (currentAnswer.trim().length < 3) {
      setErrorText("Please provide a more complete answer.");
      return;
    }
    setErrorText("");
    const updated = [...agentAnswers];
    updated[currentQuestionIndex] = currentAnswer.trim();
    setAgentAnswers(updated);

    if (currentQuestionIndex < (sonicQuestions.length || 6) - 1) {
      const nextIdx = currentQuestionIndex + 1;
      setCurrentQuestionIndex(nextIdx);
      // Pre-fill sample answers if loaded
      setCurrentAnswer(updated[nextIdx] || "");
    } else {
      // All questions answered — run Sonic
      setCurrentAnswer("");
      void runSonicInterview(updated);
    }
  }

  async function runSonicInterview(answers: string[]): Promise<void> {
    setCallPhase("processing");
    setErrorText("");

    try {
      const res = await fetch(`${API_BASE_URL}/api/sonic/interview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers, language }),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(typeof body?.detail === "string" ? body.detail : "Sonic interview failed");
      }
      const result = body as SonicIntakeResponse;
      setSonicResult(result);
      setCompiledTranscript(result.transcript);
      setCallPhase("done");
      setSuccessText(
        result.success
          ? "Nova Sonic interview complete. Review the case brief and click Analyze & Route."
          : "Interview complete (fallback mode). Review and analyze."
      );
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : "Sonic interview failed.");
      setCallPhase("idle");
    }
  }

  function resetCall(): void {
    setCallPhase("idle");
    setCurrentQuestionIndex(0);
    setAgentAnswers(Array(6).fill(""));
    setCurrentAnswer("");
    setSonicResult(null);
    setCompiledTranscript("");
    setDraft(null);
    setAnalysisMeta(null);
    setErrorText("");
    setSuccessText("");
  }

  // ── Voice recording (for current answer textarea) ───────────────────────────

  function startRecording(): void {
    setErrorText("");
    if (!speechCtor) {
      setErrorText("Speech recognition is not supported in this browser.");
      return;
    }
    const rec = new speechCtor();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = language === "hi" ? "hi-IN" : "en-US";
    rec.onresult = (event) => {
      let text = "";
      for (const r of event.results) {
        if (r.isFinal) text += ` ${r[0].transcript}`;
      }
      if (text.trim()) setCurrentAnswer((prev) => `${prev} ${text}`.trim());
    };
    rec.onerror = (event) => {
      setErrorText(`Recording error: ${event.error ?? "unknown"}`);
      setIsRecording(false);
    };
    rec.onend = () => setIsRecording(false);
    recognitionRef.current = rec;
    rec.start();
    setIsRecording(true);
  }

  function stopRecording(): void {
    recognitionRef.current?.stop();
    setIsRecording(false);
  }

  // ── Nova Lite analysis ──────────────────────────────────────────────────────

  async function analyzeTranscript(): Promise<void> {
    setErrorText("");
    setSuccessText("");
    if (compiledTranscript.trim().length < 10) {
      setErrorText("Complete the Sonic interview first.");
      return;
    }
    setIsAnalyzing(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: compiledTranscript, language }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : "Analysis failed");
      const data = body as AnalyzeResponse;
      setDraft(data.draft);
      setAnalysisMeta({ confidence: data.confidence, source: data.source, warnings: data.warnings });
      setSuccessText("Nova Lite generated the escalation decision. Review and approve below.");
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function createTicket(): Promise<void> {
    if (!draft) { setErrorText("Analyze the case first."); return; }
    setErrorText("");
    setSuccessText("");
    setIsCreating(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/tickets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: compiledTranscript, draft }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : "Ticket creation failed");
      const ticket = body as TicketRecord;
      setSuccessText(`✓ Ticket ${ticket.id} routed to ${ticket.draft.route_to}.`);
      setTickets((prev) => [ticket, ...prev].slice(0, 10));
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : "Ticket creation failed.");
    } finally {
      setIsCreating(false);
    }
  }

  function updateDraftField<K extends keyof TicketDraft>(key: K, value: TicketDraft[K]): void {
    setDraft((prev) => prev ? { ...prev, [key]: value } : prev);
  }

  function riskTone(level: RiskLevel | TicketPriority | EscalationLevel): string {
    if (level === "critical" || level === "incident" || level === "high" || level === "specialist") return "tone-high";
    if (level === "medium" || level === "supervisor") return "tone-medium";
    return "tone-low";
  }

  const confidencePercent = analysisMeta ? Math.round(analysisMeta.confidence * 100) : 0;
  const questions = sonicQuestions.length ? sonicQuestions : SAMPLE_ANSWERS.map((_, i) => ({
    key: `q${i}`,
    label: ["Issue Summary", "Reference Details", "Timeline", "Customer Impact", "Urgency", "Desired Resolution"][i],
    spoken: ["What exactly is the customer reporting?", "Which order or account is affected?", "When did this start?", "How is this impacting the customer?", "Is there deadline or escalation risk?", "What outcome does the customer want?"][i],
  }));
  const totalQuestions = questions.length;
  const currentQ = questions[currentQuestionIndex];

  return (
    <div className="page-shell">
      <div className="orb orb-one" aria-hidden />
      <div className="orb orb-two" aria-hidden />

      {/* ── Hero ── */}
      <header className="hero">
        <p className="badge">Amazon Nova Hackathon · #AmazonNova</p>
        <h1>Voice Escalation Copilot</h1>
        <p>
          Powered by <strong>Amazon Nova Sonic</strong> (speech-to-speech interview) +{" "}
          <strong>Nova Lite</strong> (escalation routing). Speak the case — Nova handles the rest.
        </p>
      </header>

      {/* ── Queue metrics ── */}
      <section className="manager-strip">
        <div className="metric-card">
          <span>Open Queue</span>
          <strong>{queueMetrics.openCount}</strong>
        </div>
        <div className="metric-card">
          <span>Manager Review</span>
          <strong>{queueMetrics.managerReview}</strong>
        </div>
        <div className="metric-card">
          <span>Critical Cases</span>
          <strong>{queueMetrics.criticalCount}</strong>
        </div>
        <div className="metric-card">
          <span>SLA At Risk</span>
          <strong>{queueMetrics.slaHot}</strong>
        </div>
      </section>

      <main className="grid-layout">
        {/* ── Nova Sonic Interview Panel ── */}
        <section className="panel entry-panel">
          <div className="panel-header">
            <div>
              <h2>Nova Sonic Interview</h2>
              <p className="panel-sub">
                Nova Sonic conducts a structured 6-question intake interview. Answer each question — by voice or by
                typing — then Nova Lite routes the case.
              </p>
            </div>
            <div className="sonic-badge">
              <span className={`sonic-dot ${callPhase === "questions" || callPhase === "processing" ? "sonic-dot--live" : ""}`} />
              {callPhase === "idle" && "Ready"}
              {callPhase === "questions" && "Live Interview"}
              {callPhase === "processing" && "Processing…"}
              {callPhase === "done" && "Complete"}
            </div>
          </div>

          {/* Language selector */}
          <div className="row two-cols" style={{ marginBottom: "0.6rem" }}>
            <label>
              Language
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="en">English</option>
                <option value="hi">Hindi</option>
              </select>
            </label>
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end", gap: "0.35rem" }}>
              <span style={{ fontSize: "0.82rem", fontWeight: 500, color: "#5b5d66" }}>Quick Start</span>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button type="button" onClick={startCall} disabled={callPhase === "processing"} id="btn-start-call">
                  Start Interview
                </button>
                <button type="button" className="ghost" onClick={loadSampleAnswers} disabled={callPhase === "processing"} id="btn-load-sample">
                  Load Demo Case
                </button>
              </div>
            </div>
          </div>

          {/* ── Progress bar ── */}
          {callPhase !== "idle" && (
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: callPhase === "done"
                    ? "100%"
                    : `${Math.round((currentQuestionIndex / totalQuestions) * 100)}%`,
                }}
              />
              <span className="progress-label">
                {callPhase === "done"
                  ? "All 6 questions answered"
                  : `Question ${currentQuestionIndex + 1} of ${totalQuestions}`}
              </span>
            </div>
          )}

          {/* ── Active question card ── */}
          {callPhase === "questions" && currentQ && (
            <div className="sonic-question-card">
              <div className="sonic-avatar">
                <Waveform active={isRecording} />
                <span className="sonic-model-tag">Nova Sonic</span>
              </div>
              <div className="sonic-bubble">
                <span className="bubble-step">{currentQ.label}</span>
                <p className="bubble-text">{currentQ.spoken}</p>
              </div>
              <label className="answer-label">
                Your Answer
                <textarea
                  id="txt-answer"
                  value={currentAnswer}
                  onChange={(e) => setCurrentAnswer(e.target.value)}
                  placeholder="Type your answer or click the mic to speak…"
                />
              </label>
              <div className="actions-row">
                <button type="button" id="btn-save-answer" onClick={saveAnswer}>
                  {currentQuestionIndex < totalQuestions - 1 ? "Save & Next →" : "Submit to Nova Sonic"}
                </button>
                <button
                  type="button"
                  className={`ghost mic-btn ${isRecording ? "mic-btn--active" : ""}`}
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!speechCtor}
                  id="btn-mic"
                >
                  {isRecording ? "⏹ Stop Mic" : "🎙 Speak"}
                </button>
              </div>
            </div>
          )}

          {/* ── Processing state ── */}
          {callPhase === "processing" && (
            <div className="sonic-processing-card">
              <Waveform active />
              <p className="processing-headline">Nova Sonic is conducting the interview…</p>
              <NovaThinking />
              <p className="processing-sub">
                The bidirectional Sonic stream is replaying your 6 answers through the model and
                generating structured acknowledgements.
              </p>
            </div>
          )}

          {/* ── Interview complete ── */}
          {callPhase === "done" && sonicResult && (
            <div className="sonic-complete-card">
              <div className="complete-header">
                <span className="complete-icon">✓</span>
                <div>
                  <strong>Nova Sonic Interview Complete</strong>
                  <p>
                    {sonicResult.success
                      ? "Processed by amazon.nova-2-sonic-v1:0"
                      : "Processed in fallback mode"}
                  </p>
                </div>
              </div>
              <div className="answer-review">
                {sonicResult.answers.map((a, i) => (
                  <div key={a.key} className="answer-review-item">
                    <div className="review-q">
                      <span className="review-step">Q{i + 1}</span>
                      <strong>{a.label}</strong>
                    </div>
                    <p className="review-answer">{a.answer}</p>
                    {a.nova_ack && (
                      <p className="review-ack">
                        <span className="ack-label">Nova:</span> {a.nova_ack}
                      </p>
                    )}
                  </div>
                ))}
              </div>
              <label style={{ marginTop: "0.8rem" }}>
                Compiled Case Brief
                <textarea
                  id="txt-transcript"
                  value={compiledTranscript}
                  onChange={(e) => setCompiledTranscript(e.target.value)}
                  style={{ minHeight: "130px" }}
                />
              </label>
            </div>
          )}

          {/* Bottom actions */}
          <div className="actions-row" style={{ marginTop: "0.8rem" }}>
            {callPhase === "done" && (
              <button
                type="button"
                id="btn-analyze"
                onClick={analyzeTranscript}
                disabled={isAnalyzing || !compiledTranscript}
              >
                {isAnalyzing ? <><NovaThinking /> Analyzing…</> : "⚡ Analyze & Route with Nova Lite"}
              </button>
            )}
            {callPhase !== "idle" && (
              <button type="button" className="ghost" id="btn-reset" onClick={resetCall}>
                Reset
              </button>
            )}
          </div>
        </section>

        {/* ── Decision Panel ── */}
        <section className="panel decision-panel">
          <h2>Escalation Decision</h2>
          <p className="panel-sub">
            Nova Lite uses the full Sonic case brief to decide routing, urgency, risk signals, and next actions.
          </p>

          {draft ? (
            <div className="draft-form">
              <div className="meta-strip">
                <span>Source: {analysisMeta?.source ?? "-"}</span>
                <span>Confidence: {confidencePercent}%</span>
                <span className="nova-powered-tag">⚡ Powered by Amazon Nova</span>
              </div>

              <div className="signal-grid">
                <div className="signal-card">
                  <span>Route To</span>
                  <strong>{draft.route_to}</strong>
                </div>
                <div className={`signal-card ${riskTone(draft.escalation_level)}`}>
                  <span>Escalation</span>
                  <strong>{draft.escalation_level}</strong>
                </div>
                <div className={`signal-card ${riskTone(draft.sla_risk)}`}>
                  <span>SLA Risk</span>
                  <strong>{draft.sla_risk}</strong>
                </div>
                <div className={`signal-card ${riskTone(draft.churn_risk)}`}>
                  <span>Churn Risk</span>
                  <strong>{draft.churn_risk}</strong>
                </div>
              </div>

              <div className="decision-banner">
                <div>
                  <span>Manager Review</span>
                  <strong>{draft.requires_manager_review ? "Required" : "Not Required"}</strong>
                </div>
                <div>
                  <span>Refund Signal</span>
                  <strong>{draft.refund_requested ? "Detected" : "No"}</strong>
                </div>
              </div>

              <label>
                Title
                <input id="inp-title" value={draft.title} onChange={(e) => updateDraftField("title", e.target.value)} />
              </label>
              <label>
                Summary
                <textarea value={draft.summary} onChange={(e) => updateDraftField("summary", e.target.value)} />
              </label>
              <div className="row two-cols">
                <label>
                  Category
                  <select value={draft.category} onChange={(e) => updateDraftField("category", e.target.value as TicketCategory)}>
                    <option value="billing">Billing</option>
                    <option value="delivery">Delivery</option>
                    <option value="technical">Technical</option>
                    <option value="account">Account</option>
                    <option value="refund">Refund</option>
                    <option value="other">Other</option>
                  </select>
                </label>
                <label>
                  Priority
                  <select value={draft.priority} onChange={(e) => updateDraftField("priority", e.target.value as TicketPriority)}>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </label>
              </div>
              <label>
                Customer Impact
                <textarea value={draft.customer_impact} onChange={(e) => updateDraftField("customer_impact", e.target.value)} />
              </label>
              <label>
                Resolution Path
                <textarea value={draft.resolution_path} onChange={(e) => updateDraftField("resolution_path", e.target.value)} />
              </label>
              <label>
                Next Actions (one per line)
                <textarea
                  value={draft.next_actions.join("\n")}
                  onChange={(e) =>
                    updateDraftField("next_actions", e.target.value.split("\n").map((s) => s.trim()).filter(Boolean))
                  }
                />
              </label>
              <label>
                Suggested Reply
                <textarea value={draft.suggested_reply} onChange={(e) => updateDraftField("suggested_reply", e.target.value)} />
              </label>

              {analysisMeta?.warnings?.length ? (
                <div className="warning-box">
                  {analysisMeta.warnings.map((w) => <p key={w}>{w}</p>)}
                </div>
              ) : null}

              <button type="button" id="btn-create-ticket" onClick={createTicket} disabled={isCreating}>
                {isCreating ? "Routing…" : "Approve & Create Ticket"}
              </button>
            </div>
          ) : (
            <div className="empty-state">
              Complete the Nova Sonic interview, then click <strong>Analyze &amp; Route</strong> to generate a
              routing decision here.
            </div>
          )}
        </section>
      </main>

      {/* ── Status messages ── */}
      {(errorText || successText) && (
        <section className="status-row">
          {errorText ? <p className="error">{errorText}</p> : null}
          {successText ? <p className="success">{successText}</p> : null}
        </section>
      )}

      {/* ── Escalation Queue ── */}
      <section className="panel tickets-panel">
        <div className="tickets-head">
          <div>
            <h2>Escalation Queue</h2>
            <p className="panel-sub">Recent routed tickets for supervisors and support managers.</p>
          </div>
          <button type="button" className="ghost" id="btn-refresh" onClick={() => void fetchTickets()} disabled={isLoadingTickets}>
            {isLoadingTickets ? "Refreshing…" : "Refresh"}
          </button>
        </div>

        {tickets.length ? (
          <div className="ticket-list">
            {tickets.map((ticket) => (
              <article key={ticket.id} className="ticket-item">
                <div className="ticket-main">
                  <div>
                    <h3>{ticket.draft.title}</h3>
                    <p>{ticket.draft.summary}</p>
                  </div>
                  <div className="ticket-route">{ticket.draft.route_to}</div>
                </div>
                <div className="ticket-tags">
                  <span>{ticket.id}</span>
                  <span>{ticket.draft.category}</span>
                  <span className={riskTone(ticket.draft.priority)}>{ticket.draft.priority}</span>
                  <span className={riskTone(ticket.draft.sla_risk)}>sla {ticket.draft.sla_risk}</span>
                  <span className={riskTone(ticket.draft.churn_risk)}>churn {ticket.draft.churn_risk}</span>
                  <span>{ticket.draft.escalation_level}</span>
                  {ticket.draft.requires_manager_review ? <span>manager review</span> : null}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-state">No routed tickets yet. Create one from the escalation decision panel.</p>
        )}
      </section>
    </div>
  );
}

export default App;
