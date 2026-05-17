"use client";

// src/components/ChatInterface.tsx

import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, ChevronRight, Loader2, Mic, MicOff, Send } from "lucide-react";
import { interact, pollTrace, TraceRow } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Monologue strings streamed in as the agent thinks */
  thoughts: string[];
  /** While true, the skeleton + spinner render instead of content */
  loading: boolean;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

/**
 * Extracts the monologues from all trace rows so far.
 * Filters null/empty strings so the UI never renders blank thought lines.
 */
function extractThoughts(rows: TraceRow[]): string[] {
  return rows
    .map((r) => r.monologue)
    .filter((m): m is string => typeof m === "string" && m.trim().length > 0);
}

/**
 * Extracts the final user-facing reply from the completed trace.
 * Walks the rows in reverse looking for `processing_finished` first,
 * then falls back to the last row's output_data.
 */
function extractFinalResponse(rows: TraceRow[]): string {
  const finished = [...rows]
    .reverse()
    .find((r) => r.step_name === "processing_finished");

  const od = finished?.output_data ?? rows.at(-1)?.output_data ?? null;
  if (!od) return "Done — but no response was generated.";

  return typeof od["final_response"] === "string"
    ? od["final_response"]
    : JSON.stringify(od, null, 2);
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ThoughtsPanel({ thoughts }: { thoughts: string[] }) {
  const [open, setOpen] = useState(false);

  if (thoughts.length === 0) return null;

  return (
    <div className="mt-2 rounded-md border border-zinc-700/50 bg-zinc-900/40 text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-zinc-400 hover:text-zinc-200 transition-colors"
      >
        <Brain size={12} className="shrink-0 text-emerald-400" />
        <span className="font-mono tracking-wide">
          {thoughts.length} thought{thoughts.length > 1 ? "s" : ""} captured
        </span>
        <span className="ml-auto">
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
      </button>

      {open && (
        <div className="border-t border-zinc-700/50 px-3 py-2 space-y-2 max-h-48 overflow-y-auto">
          {thoughts.map((t, i) => (
            <p
              key={i}
              className="font-mono text-zinc-400 whitespace-pre-wrap leading-relaxed"
            >
              <span className="text-emerald-500/70 select-none mr-1">▸</span>
              {t}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: Message }) {
  return (
    <div className="flex flex-col items-start max-w-[80%]">
      {/* Agent label */}
      <span className="mb-1 px-1 text-[10px] font-mono tracking-widest uppercase text-zinc-500">
        butler
      </span>

      <div className="rounded-2xl rounded-tl-sm bg-zinc-800/70 border border-zinc-700/40 px-4 py-3 text-zinc-100 leading-relaxed shadow-md backdrop-blur-sm">
        {msg.loading ? (
          <div className="flex items-center gap-2 text-zinc-400">
            <Loader2 size={14} className="animate-spin text-emerald-400" />
            <span className="font-mono text-xs tracking-wide animate-pulse">
              processing…
            </span>
          </div>
        ) : (
          <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
        )}
      </div>

      <ThoughtsPanel thoughts={msg.thoughts} />
    </div>
  );
}

function UserBubble({ msg }: { msg: Message }) {
  return (
    <div className="flex flex-col items-end max-w-[80%] self-end">
      <span className="mb-1 px-1 text-[10px] font-mono tracking-widest uppercase text-zinc-500">
        you
      </span>
      <div className="rounded-2xl rounded-tr-sm bg-emerald-500/10 border border-emerald-500/20 px-4 py-3 text-zinc-100 text-sm leading-relaxed shadow-md backdrop-blur-sm">
        <p className="whitespace-pre-wrap">{msg.content}</p>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "assistant",
      content:
        "System online. I'm your local AI butler — ask me to schedule something, search your notes, or just chat.",
      thoughts: [],
      loading: false,
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Voice state ───────────────────────────────────────────────────────────
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  /**
   * Whether the current browser exposes any SpeechRecognition constructor.
   * Evaluated lazily so SSR never touches `window`.
   */
  const sttSupported =
    typeof window !== "undefined" &&
    ("SpeechRecognition" in window || "webkitSpeechRecognition" in window);

  const ttsSupported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  /**
   * Speak `text` via the Web Speech API.
   *
   * Voice-selection strategy (in priority order):
   *   1. Microsoft voices  (Edge / Windows — highest quality)
   *   2. Google voices     (Chrome on macOS / Linux)
   *   3. Any `en` voice    (Safari, Firefox, etc.)
   *   4. Browser default   (no preference, still speaks)
   *
   * We wait for `voiceschanged` because Chrome loads voices asynchronously
   * and `getVoices()` returns [] on the first synchronous call.
   */
  const speak = useCallback(
    (text: string) => {
      if (!ttsSupported) return;

      // Cancel any in-flight utterance so a rapid new response doesn't queue.
      window.speechSynthesis.cancel();

      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0;
      utter.pitch = 1.0;

      const selectVoiceAndSpeak = () => {
        const voices = window.speechSynthesis.getVoices();

        const preferred =
          voices.find((v) => /microsoft/i.test(v.name) && v.lang.startsWith("en")) ??
          voices.find((v) => /google/i.test(v.name) && v.lang.startsWith("en")) ??
          voices.find((v) => v.lang.startsWith("en")) ??
          null;

        if (preferred) utter.voice = preferred;
        window.speechSynthesis.speak(utter);
      };

      // Voices already loaded (Firefox, Edge, cached Chrome)
      if (window.speechSynthesis.getVoices().length > 0) {
        selectVoiceAndSpeak();
      } else {
        // Chrome: wait for the async voices-loaded event, then speak once.
        const onVoicesChanged = () => {
          window.speechSynthesis.removeEventListener("voiceschanged", onVoicesChanged);
          selectVoiceAndSpeak();
        };
        window.speechSynthesis.addEventListener("voiceschanged", onVoicesChanged);
      }
    },
    [ttsSupported]
  );

  /**
   * Toggle microphone listening.
   *
   * `continuous: false`  — fires `onresult` after a natural pause, which is
   *   the right UX for command-style input (vs. dictation mode).
   * `interimResults: true` — updates the textarea in real time so the user
   *   sees words appearing as they speak, not just after they stop.
   */
  const toggleListening = useCallback(() => {
    if (!sttSupported) return;

    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    // Stop any active TTS so the mic doesn't pick up the butler's voice.
    if (ttsSupported) window.speechSynthesis.cancel();

    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition ??
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) return;

    const recognition = new SpeechRecognitionCtor();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;

    // Track the stable (final) transcript prefix so interim results
    // don't duplicate already-committed words.
    let committedTranscript = "";

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          committedTranscript += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      // Replace only the live-draft portion; preserve whatever the user
      // had already typed before they clicked the mic.
      setInput((prev) => {
        const base = prev.trimEnd();
        const separator = base.length > 0 ? " " : "";
        return base + separator + committedTranscript + interim;
      });
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // `no-speech` is benign (user just didn't say anything); don't surface it.
      if (event.error !== "no-speech") {
        setError(`Microphone error: ${event.error}`);
      }
      setListening(false);
    };

    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }, [listening, sttSupported, ttsSupported]);

  // Auto-scroll to bottom on new messages or thought updates
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cleanup interval on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (traceId: string, assistantMsgId: string) => {
      pollRef.current = setInterval(async () => {
        const result = await pollTrace(traceId);

        if (result.status === "queued") {
          // Still in queue — no rows yet, show processing pulse
          return;
        }

        if (result.status === "error") {
          stopPolling();
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, loading: false, content: `⚠ Error: ${result.message}` }
                : m
            )
          );
          setBusy(false);
          return;
        }

        // status === "ok" — update thoughts in real time
        const thoughts = extractThoughts(result.data.rows);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, thoughts } : m
          )
        );

        if (result.data.completed) {
          stopPolling();
          const finalResponse = extractFinalResponse(result.data.rows);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, loading: false, content: finalResponse, thoughts }
                : m
            )
          );
          setBusy(false);
          // ── TTS: read the response aloud ──────────────────────────────
          speak(finalResponse);
          // Refocus input after completion
          setTimeout(() => inputRef.current?.focus(), 50);
        }
      }, 2000);
    },
    [stopPolling, speak]
  );

  const handleSubmit = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || busy) return;

    setError(null);
    setInput("");
    setBusy(true);

    // Append user message immediately
    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: trimmed,
      thoughts: [],
      loading: false,
    };

    // Append skeleton assistant message
    const assistantMsgId = uid();
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      thoughts: [],
      loading: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      const { trace_id } = await interact(trimmed);
      startPolling(trace_id, assistantMsgId);
    } catch (err) {
      stopPolling();
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, loading: false, content: `⚠ Failed to reach server: ${msg}` }
            : m
        )
      );
      setBusy(false);
    }
  }, [input, busy, startPolling, stopPolling]);

  // Cmd/Ctrl+Enter or Enter (without shift) submits
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 font-['IBM_Plex_Mono',monospace]">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center gap-3 px-6 py-4 border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.4)]" />
          <span className="text-xs tracking-[0.2em] uppercase text-zinc-300">
            AI ButlerOS
          </span>
        </div>
        <span className="ml-auto text-[10px] text-zinc-600 tracking-widest uppercase">
          v0.6 · local
        </span>
      </header>

      {/* ── Message list ────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        <div className="mx-auto w-full max-w-2xl flex flex-col gap-6">
          {messages.map((msg) =>
            msg.role === "user" ? (
              <UserBubble key={msg.id} msg={msg} />
            ) : (
              <AssistantBubble key={msg.id} msg={msg} />
            )
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── Error banner ─────────────────────────────────────────────────────── */}
      {error && (
        <div className="mx-auto w-full max-w-2xl px-4 pb-2">
          <p className="rounded-md border border-red-500/30 bg-red-950/30 px-3 py-2 text-xs text-red-400 font-mono">
            {error}
          </p>
        </div>
      )}

      {/* ── Input bar ───────────────────────────────────────────────────────── */}
      <footer className="shrink-0 px-4 pb-6 pt-2">
        <div className="mx-auto w-full max-w-2xl">
          <div className="relative flex items-end gap-2 rounded-2xl border border-zinc-700/60 bg-zinc-900/80 px-4 py-3 shadow-xl backdrop-blur-sm focus-within:border-emerald-500/40 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything…"
              rows={1}
              disabled={busy}
              className="flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder:text-zinc-600 outline-none leading-relaxed max-h-40 overflow-y-auto disabled:opacity-40"
              style={{ fieldSizing: "content" } as React.CSSProperties}
            />

            <button
              onClick={toggleListening}
              disabled={busy || !sttSupported}
              aria-label={listening ? "Stop listening" : "Start voice input"}
              title={
                !sttSupported
                  ? "Speech recognition not supported in this browser"
                  : listening
                  ? "Stop listening"
                  : "Speak your message"
              }
              className={[
                "mb-0.5 shrink-0 flex items-center justify-center h-8 w-8 rounded-xl transition-all",
                "disabled:opacity-30 disabled:pointer-events-none active:scale-95",
                listening
                  ? // Glowing red ring when hot
                    "bg-red-500/15 border border-red-500/60 text-red-400 shadow-[0_0_10px_2px_rgba(239,68,68,0.3)] animate-pulse"
                  : "bg-zinc-700/60 text-zinc-400 hover:bg-zinc-600/70 hover:text-zinc-200",
              ].join(" ")}
            >
              {listening ? <MicOff size={14} /> : <Mic size={14} />}
            </button>

            <button
              onClick={handleSubmit}
              disabled={busy || !input.trim()}
              aria-label="Send message"
              className="mb-0.5 shrink-0 flex items-center justify-center h-8 w-8 rounded-xl bg-emerald-500 text-zinc-950 shadow-md hover:bg-emerald-400 active:scale-95 transition-all disabled:opacity-30 disabled:pointer-events-none"
            >
              {busy ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Send size={14} />
              )}
            </button>
          </div>

          <p className="mt-2 text-center text-[10px] text-zinc-700 tracking-widest uppercase">
            Enter to send · Shift+Enter for new line · Mic for voice
          </p>
        </div>
      </footer>
    </div>
  );
}