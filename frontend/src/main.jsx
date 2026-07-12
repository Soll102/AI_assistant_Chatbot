import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bot, ChevronLeft, FileText, Loader2, MessageSquarePlus, Send, Trash2, Upload } from "lucide-react";
import katex from "katex";
import { marked } from "marked";
import "katex/dist/katex.min.css";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const MAX_QUESTION_LENGTH = 500;
const WELCOME_MESSAGE = {
  role: "assistant",
  content: "",
  sources: [],
};

function App() {
  const [documents, setDocuments] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [question, setQuestion] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [previewPage, setPreviewPage] = useState(1);
  const [isPreviewHidden, setIsPreviewHidden] = useState(false);
  const [isPreviewGuardActive, setIsPreviewGuardActive] = useState(true);
  const [historyMenu, setHistoryMenu] = useState(null);
  const [documentMenu, setDocumentMenu] = useState(null);
  const [previewMenu, setPreviewMenu] = useState(null);
  const [panelSizes, setPanelSizes] = useState(() => {
    const saved = localStorage.getItem("rag-panel-sizes");
    return saved ? normalizePanelSizes(JSON.parse(saved)) : { sidebar: 250, preview: 560 };
  });
  const fileInputRef = useRef(null);

  const activeDocument = useMemo(
    () => documents.find((item) => item.id === activeDocumentId),
    [documents, activeDocumentId],
  );

  const pdfUrl = activeDocumentId
    ? `${API_BASE}/api/documents/${activeDocumentId}/file#page=${previewPage}&view=FitH`
    : "";

  useEffect(() => {
    loadDocuments();
    loadSessions();
  }, []);

  useEffect(() => {
    localStorage.setItem("rag-panel-sizes", JSON.stringify(panelSizes));
  }, [panelSizes]);

  useEffect(() => {
    function closeContextMenus() {
      setHistoryMenu(null);
      setDocumentMenu(null);
      setPreviewMenu(null);
    }
    window.addEventListener("click", closeContextMenus);
    window.addEventListener("keydown", closeContextMenus);
    return () => {
      window.removeEventListener("click", closeContextMenus);
      window.removeEventListener("keydown", closeContextMenus);
    };
  }, []);

  async function loadDocuments() {
    const response = await fetch(`${API_BASE}/api/documents`);
    if (response.ok) {
      const data = await response.json();
      setDocuments(data);
      setActiveDocumentId((current) => current || data[0]?.id || "");
    }
  }

  async function loadSessions() {
    const response = await fetch(`${API_BASE}/api/chat/sessions`);
    if (response.ok) {
      setSessions(await response.json());
    }
  }

  async function createNewChat() {
    const response = await fetch(`${API_BASE}/api/chat/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: "Chat mới",
        document_id: activeDocumentId || null,
      }),
    });
    if (response.ok) {
      const session = await response.json();
      setActiveSessionId(session.id);
      setMessages([WELCOME_MESSAGE]);
      await loadSessions();
    }
  }

  async function openSession(sessionId) {
    setHistoryMenu(null);
    setActiveSessionId(sessionId);
    const response = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}/messages`);
    if (response.ok) {
      const data = await response.json();
      setMessages(
        data.length
          ? data.map((message) => ({ role: message.role, content: message.content, sources: [] }))
          : [WELCOME_MESSAGE],
      );
    }
  }

  async function deleteSession(sessionId) {
    setHistoryMenu(null);
    const response = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, { method: "DELETE" });
    if (!response.ok) return;

    setSessions((current) => current.filter((session) => session.id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId("");
      setMessages([WELCOME_MESSAGE]);
    }
  }

  async function deleteDocument(documentId) {
    setDocumentMenu(null);
    const response = await fetch(`${API_BASE}/api/documents/${documentId}`, { method: "DELETE" });
    if (!response.ok) return;

    setDocuments((current) => {
      const nextDocuments = current.filter((document) => document.id !== documentId);
      if (activeDocumentId === documentId) {
        setActiveDocumentId(nextDocuments[0]?.id || "");
        setPreviewPage(1);
        setMessages([WELCOME_MESSAGE]);
      }
      return nextDocuments;
    });
  }

  function startResize(handle) {
    return (event) => {
      event.preventDefault();
      const startX = event.clientX;
      const startSizes = { ...panelSizes };

      function onMove(moveEvent) {
        const delta = moveEvent.clientX - startX;
        setPanelSizes(() => {
          if (handle === "sidebar") {
            return {
              ...startSizes,
              sidebar: clamp(startSizes.sidebar + delta, 200, 420),
            };
          }
          return {
            ...startSizes,
            preview: clamp(startSizes.preview - delta, 260, Math.max(360, window.innerWidth - 540)),
          };
        });
      }

      function onUp() {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        document.body.classList.remove("resizing");
      }

      document.body.classList.add("resizing");
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    };
  }

  async function uploadPdf(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/api/documents`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Upload thất bại.");
      }
      setDocuments((current) => [data, ...current.filter((item) => item.id !== data.id)]);
      setActiveDocumentId(data.id);
      setPreviewPage(1);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Đã index **${data.filename}**: ${data.pages} trang, ${data.chunks} chunks. Bạn có thể hỏi về tài liệu này rồi.`,
          sources: [],
        },
      ]);
      await loadSessions();
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Upload lỗi: ${error.message}`, sources: [] },
      ]);
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || isAsking) return;

    setQuestion("");
    setIsAsking(true);
    setMessages((current) => [...current, { role: "user", content: cleanQuestion, sources: [] }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: cleanQuestion,
          document_id: activeDocumentId || null,
          session_id: activeSessionId || null,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Chat thất bại.");
      }
      if (data.session_id) {
        setActiveSessionId(data.session_id);
      }
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
          toolName: data.tool_name,
          verification: data.verification,
        },
      ]);
      await loadSessions();
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Chat lỗi: ${error.message}`, sources: [] },
      ]);
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main
      className="app-shell"
      style={{
        gridTemplateColumns: isPreviewHidden
          ? `${panelSizes.sidebar}px 10px minmax(280px, 1fr)`
          : `${panelSizes.sidebar}px 10px minmax(280px, 1fr) 10px ${panelSizes.preview}px`,
      }}
    >
      <aside className="sidebar">
        <div className="brand">
          <Bot size={22} />
          <div>
            <strong>AI Chat Bot</strong>
          </div>
        </div>

        <button className="primary-button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
          {isUploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
          {isUploading ? "Đang index..." : "Upload PDF"}
        </button>
        <input ref={fileInputRef} type="file" accept="application/pdf" hidden onChange={uploadPdf} />

        <section className="document-list">
          <h2>Tài liệu</h2>
          {documents.length === 0 ? (
            <p className="muted">Chưa có PDF nào.</p>
          ) : (
            documents.map((document) => (
              <div className="document-row" key={document.id}>
                <button
                  className={`document-item ${document.id === activeDocumentId ? "active" : ""}`}
                  onClick={() => {
                    setActiveDocumentId(document.id);
                    setPreviewPage(1);
                  }}
                  onContextMenu={(event) => {
                    event.preventDefault();
                    setDocumentMenu({ documentId: document.id });
                  }}
                >
                  <FileText size={18} />
                  <span>
                    <strong>{document.filename}</strong>
                    <small>
                      {document.pages} trang · {document.chunks} chunks
                    </small>
                  </span>
                </button>
                {documentMenu?.documentId === document.id && (
                  <button
                    className="document-delete"
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteDocument(document.id);
                    }}
                    title="Xoá PDF"
                  >
                    <Trash2 size={14} />
                    Xoá
                  </button>
                )}
              </div>
            ))
          )}
        </section>

        <section className="history-list">
          <h2>Lịch sử chat</h2>
          {sessions.length === 0 ? (
            <p className="muted">Chưa có lịch sử.</p>
          ) : (
            sessions.map((session) => (
              <div className="history-row" key={session.id}>
                <button
                  className={`history-item ${session.id === activeSessionId ? "active" : ""}`}
                  onClick={() => openSession(session.id)}
                  onContextMenu={(event) => {
                    event.preventDefault();
                    setHistoryMenu({ sessionId: session.id });
                  }}
                >
                  <strong>{session.title}</strong>
                  <small>{new Date(session.updated_at).toLocaleString("vi-VN")}</small>
                </button>
                {historyMenu?.sessionId === session.id && (
                  <button
                    className="history-delete"
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      deleteSession(session.id);
                    }}
                    title="Xoá đoạn chat"
                  >
                    <Trash2 size={14} />
                    Xoá
                  </button>
                )}
              </div>
            ))
          )}
        </section>
      </aside>

      <div className="resize-handle" onMouseDown={startResize("sidebar")} />

      <section className="chat-panel">
        <header className="panel-header">
          <div>
            <span className="eyebrow">Chat</span>
          </div>
          <button className="ghost-button" onClick={createNewChat}>
            <MessageSquarePlus size={17} />
            Chat mới
          </button>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div
                className="message-body"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
              />
              {(message.toolName || message.verification) && (
                <div className="message-meta">
                  {message.toolName && <span>Tool: {message.toolName}</span>}
                  {message.verification && <span>Verification: {message.verification}</span>}
                </div>
              )}
              {message.sources?.length > 0 && (
                <SourceList
                  sources={message.sources}
                  onOpenSource={(source) => {
                    setActiveDocumentId(source.document_id);
                    setPreviewPage(source.preview_page || source.page);
                  }}
                />
              )}
            </article>
          ))}
          {isAsking && (
            <article className="message assistant">
              <div className="typing">
                <Loader2 className="spin" size={16} />
                Đang truy xuất tài liệu...
              </div>
            </article>
          )}
        </div>

        <form className="composer" onSubmit={askQuestion}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value.slice(0, MAX_QUESTION_LENGTH))}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder={activeDocument ? "Hỏi về PDF này..." : "Upload hoặc chọn PDF trước..."}
            maxLength={MAX_QUESTION_LENGTH}
            rows={1}
          />
          <span className={`char-counter ${question.length >= MAX_QUESTION_LENGTH ? "limit" : ""}`}>
            {question.length}/{MAX_QUESTION_LENGTH}
          </span>
          <button type="submit" disabled={!question.trim() || isAsking}>
            <Send size={18} />
          </button>
        </form>
      </section>

      {!isPreviewHidden && <div className="resize-handle" onMouseDown={startResize("preview")} />}

      {!isPreviewHidden ? (
        <aside
          className="preview-panel"
          onMouseLeave={() => setIsPreviewGuardActive(true)}
          onContextMenu={(event) => {
            event.preventDefault();
            setPreviewMenu({ x: event.clientX, y: event.clientY });
          }}
        >
          {isPreviewGuardActive && (
            <div
              className="preview-context-guard"
              onContextMenu={(event) => {
                event.preventDefault();
                setPreviewMenu({ x: event.clientX, y: event.clientY });
              }}
              onMouseDown={(event) => {
                if (event.button !== 2) {
                  setIsPreviewGuardActive(false);
                }
              }}
              onWheel={() => setIsPreviewGuardActive(false)}
            />
          )}
          {previewMenu && (
            <button
              className="preview-context-action"
              type="button"
              style={{ left: previewMenu.x, top: previewMenu.y }}
              onClick={(event) => {
                event.stopPropagation();
                setPreviewMenu(null);
                setIsPreviewHidden(true);
              }}
            >
              Ẩn preview
            </button>
          )}
          {pdfUrl ? (
            <iframe
              key={`${activeDocumentId}-${previewPage}`}
              className="pdf-frame"
              src={pdfUrl}
              title={`PDF preview trang ${previewPage}`}
            />
          ) : (
            <div className="empty-preview">Upload PDF để xem preview ở đây.</div>
          )}
        </aside>
      ) : (
        <button className="preview-toggle open" type="button" onClick={() => setIsPreviewHidden(false)} title="Mở preview">
          <ChevronLeft size={18} />
        </button>
      )}
    </main>
  );
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizePanelSizes(sizes) {
  return {
    sidebar: clamp(Number(sizes.sidebar) || 250, 200, 420),
    preview: clamp(Number(sizes.preview) || 560, 260, Math.max(360, window.innerWidth - 540)),
  };
}

function renderMarkdown(content) {
  return marked.parse(formatMath(content));
}

function formatMath(content) {
  const renderedMath = [];
  const stashMath = (html) => {
    const token = `@@MATH_${renderedMath.length}@@`;
    renderedMath.push(html);
    return token;
  };

  return content
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, expression) => {
      return stashMath(renderMath(expression, true));
    })
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, expression) => {
      return stashMath(renderMath(expression, true));
    })
    .replace(/(^|[^$])\$([^$\n]{1,160})\$(?!\$)/g, (_, prefix, expression) => {
      return `${prefix}${stashMath(renderMath(expression, false))}`;
    })
    .replace(/\\\(([^)\n]{1,160})\\\)/g, (_, expression) => {
      return stashMath(renderMath(expression, false));
    })
    .replace(/(^|\n)([^\n]*\\(?:sum|substack|frac|sqrt|neq|leq|geq|log|alpha|beta|theta|hat|bar)[^\n]*)/g, (_, prefix, expression) => {
      return `${prefix}${stashMath(renderMath(expression.trim(), true))}`;
    })
    .replace(/@@MATH_(\d+)@@/g, (_, index) => renderedMath[Number(index)] || "");
}

function renderMath(expression, displayMode) {
  const normalized = expression.trim();
  try {
    return katex.renderToString(normalized, {
      displayMode,
      throwOnError: false,
      strict: "ignore",
    });
  } catch {
    const className = displayMode ? "math-block" : "math-inline";
    return `<span class="${className}">${escapeHtml(normalized)}</span>`;
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function SourceList({ sources, onOpenSource }) {
  const visibleSources = sources.slice(0, 3);

  return (
    <div className="sources">
      <strong>Đoạn liên quan</strong>
      {visibleSources.map((source, index) => (
        <details key={`${source.document_id}-${source.page}-${index}`}>
          <summary>
            <span>Đoạn {index + 1}</span>
            <button type="button" onClick={() => onOpenSource(source)}>
              Mở trong preview
            </button>
          </summary>
          <p>{source.text}</p>
        </details>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
