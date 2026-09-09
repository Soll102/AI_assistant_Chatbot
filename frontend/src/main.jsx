import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bot, ChevronLeft, FileText, Loader2, MessageSquarePlus, Send, Trash2, Upload } from "lucide-react";
import katex from "katex";
import { marked } from "marked";
import "katex/dist/katex.min.css";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const MAX_QUESTION_LENGTH = 500;
const CHAT_CACHE_KEY = "rag-chat-cache-v1";
const MAX_CACHED_SESSIONS = 30;
const MAX_CACHED_MESSAGES = 200;

// Cache localStorage để thoát ra vào lại vẫn thấy lịch sử.
// Backend Vercel dùng /tmp ephemeral nên có thể quên session bất cứ lúc nào;
// các entry backend không còn sẽ được đánh dấu stale (chỉ xem + gửi tiếp
// sẽ tự tách sang chat mới).
function loadChatCache() {
  try {
    const raw = localStorage.getItem(CHAT_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return {
      sessions: Array.isArray(parsed.sessions) ? parsed.sessions : [],
      messagesBySession:
        parsed.messagesBySession && typeof parsed.messagesBySession === "object"
          ? parsed.messagesBySession
          : {},
      documents: Array.isArray(parsed.documents) ? parsed.documents : [],
      activeSessionId: typeof parsed.activeSessionId === "string" ? parsed.activeSessionId : "",
      activeDocumentId: typeof parsed.activeDocumentId === "string" ? parsed.activeDocumentId : "",
    };
  } catch {
    return null;
  }
}


function App() {
  const [initialCache] = useState(loadChatCache);
  const [documents, setDocuments] = useState(initialCache?.documents ?? []);
  const [sessions, setSessions] = useState(initialCache?.sessions ?? []);
  const [activeSessionId, setActiveSessionId] = useState(initialCache?.activeSessionId ?? "");
  const [activeDocumentId, setActiveDocumentId] = useState(initialCache?.activeDocumentId ?? "");
  const [messages, setMessages] = useState(
    () => initialCache?.messagesBySession?.[initialCache?.activeSessionId] ?? [],
  );
  const [messageCache, setMessageCache] = useState(initialCache?.messagesBySession ?? {});
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
  // Blob URL của file vừa upload trên máy user. Vercel serverless dùng /tmp
  // ephemeral + nhiều instance nên file backend có thể 404 ngay sau upload;
  // preview local thì luôn xem được mà không cần qua backend.
  const [localPdfUrls, setLocalPdfUrls] = useState({});

  const activeDocument = useMemo(
    () => documents.find((item) => item.id === activeDocumentId),
    [documents, activeDocumentId],
  );

  const localPdfUrl = activeDocumentId ? localPdfUrls[activeDocumentId] : "";
  const pdfUrl = localPdfUrl
    ? `${localPdfUrl}#page=${previewPage}&view=FitH`
    : activeDocumentId
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
    try {
      const trimmedSessions = sessions.filter((s) => s && s.id).slice(0, MAX_CACHED_SESSIONS);
      const keepIds = new Set(trimmedSessions.map((s) => s.id));
      const messagesBySession = {};
      for (const [sid, msgs] of Object.entries(messageCache)) {
        if (keepIds.has(sid) && Array.isArray(msgs)) messagesBySession[sid] = msgs.slice(-MAX_CACHED_MESSAGES);
      }
      // Lưu view đang mở, nhưng không ghi đè lịch sử đã lưu bằng view rỗng
      // (vd. vừa chuyển session hoặc xoá tài liệu).
      if (activeSessionId && keepIds.has(activeSessionId)) {
        if (messages.length || !(activeSessionId in messagesBySession)) {
          messagesBySession[activeSessionId] = messages.slice(-MAX_CACHED_MESSAGES);
        }
      }
      localStorage.setItem(
        CHAT_CACHE_KEY,
        JSON.stringify({
          sessions: trimmedSessions,
          messagesBySession,
          documents: documents.slice(0, 50),
          activeSessionId,
          activeDocumentId,
        }),
      );
    } catch {
      // localStorage đầy hoặc bị chặn: bỏ qua, app vẫn chạy với state memory.
    }
  }, [sessions, messageCache, messages, documents, activeSessionId, activeDocumentId]);

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
    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      if (!response.ok) return;
      const data = await response.json();
      setDocuments((current) => {
        if (!data.length) return current.length ? current.map((d) => ({ ...d, stale: true })) : current;
        const known = new Set(data.map((d) => d.id));
        return [...data, ...current.filter((d) => !known.has(d.id)).map((d) => ({ ...d, stale: true }))];
      });
      setActiveDocumentId((current) => current || data[0]?.id || "");
    } catch {
      // Backend không reachable: giữ bản lưu local.
    }
  }

  async function loadSessions() {
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions`);
      if (!response.ok) return;
      const data = await response.json();
      setSessions((current) => {
        const known = new Set(data.map((s) => s.id));
        const missing = current.filter((s) => !known.has(s.id)).map((s) => ({ ...s, stale: true }));
        return [...data, ...missing];
      });
    } catch {
      // Backend không reachable: giữ bản lưu local.
    }
  }

  async function createNewChat() {
    // Cất view hiện tại vào cache trước khi chuyển.
    setMessageCache((cache) => ({ ...cache, [activeSessionId || "__none"]: messages }));
    const response = await fetch(`${API_BASE}/api/chat/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: "Chat mới",
        document_id: activeDocument && !activeDocument.stale ? activeDocument.id : null,
      }),
    });
    if (response.ok) {
      const session = await response.json();
      setMessageCache((cache) => ({ ...cache, [session.id]: [] }));
      setActiveSessionId(session.id);
      setMessages([]);
      await loadSessions();
    }
  }

  async function openSession(sessionId) {
    setHistoryMenu(null);
    // Cất view hiện tại vào cache trước khi chuyển.
    setMessageCache((cache) => ({ ...cache, [activeSessionId || "__none"]: messages }));
    setActiveSessionId(sessionId);
    const cached = messageCache[sessionId];
    setMessages(cached ?? []);
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        const mapped = data.length
          ? data.map((message) => ({ role: message.role, content: message.content, sources: [] }))
          : [];
        setMessages(mapped);
        setMessageCache((cache) => ({ ...cache, [sessionId]: mapped }));
        setSessions((current) => current.map((s) => (s.id === sessionId ? { ...s, stale: false } : s)));
      } else if (response.status === 404) {
        // Backend đã quên session: giữ bản lưu local để xem.
        setSessions((current) => current.map((s) => (s.id === sessionId ? { ...s, stale: true } : s)));
      }
    } catch {
      // Backend không reachable: giữ bản lưu local.
    }
  }

  async function deleteSession(sessionId) {
    setHistoryMenu(null);
    try {
      const response = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, { method: "DELETE" });
      // Session đã mất trên backend (404 sau reset) thì vẫn xoá bản lưu local.
      if (!response.ok && response.status !== 404) return;
    } catch {
      return;
    }

    setMessageCache((cache) => {
      const { [sessionId]: _removed, ...rest } = cache;
      return rest;
    });
    setSessions((current) => current.filter((session) => session.id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId("");
      setMessages([]);
    }
  }

  async function deleteDocument(documentId) {
    setDocumentMenu(null);
    try {
      const response = await fetch(`${API_BASE}/api/documents/${documentId}`, { method: "DELETE" });
      // Backend đã quên file (404 sau reset) thì vẫn xoá bản lưu local.
      if (!response.ok && response.status !== 404) return;
    } catch {
      return;
    }

    setLocalPdfUrls((current) => {
      if (current[documentId]) URL.revokeObjectURL(current[documentId].split("#")[0]);
      const { [documentId]: _removed, ...rest } = current;
      return rest;
    });

    setDocuments((current) => {
      const nextDocuments = current.filter((document) => document.id !== documentId);
      if (activeDocumentId === documentId) {
        setActiveDocumentId(nextDocuments[0]?.id || "");
        setPreviewPage(1);
        setMessages([]);
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
      // Giữ blob URL để preview local, không phụ thuộc file trên backend.
      const blobUrl = URL.createObjectURL(file);
      setLocalPdfUrls((current) => ({ ...current, [data.id]: blobUrl }));
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

    const resumedSessionId = activeSessionId;
    const cacheKey = resumedSessionId || "__none";
    // Tài liệu stale (backend đã quên sau reset) thì không lọc theo id ma —
    // tìm trên toàn bộ docs backend đang có để tăng cơ hội trúng.
    const effectiveDocumentId = activeDocument && !activeDocument.stale ? activeDocument.id : null;
    const userEntry = { role: "user", content: cleanQuestion, sources: [] };
    setQuestion("");
    setIsAsking(true);
    setMessages((current) => [...current, userEntry]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: cleanQuestion,
          document_id: effectiveDocumentId,
          session_id: activeSessionId || null,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Chat thất bại.");
      }
      const assistantEntry = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        toolName: data.tool_name,
        verification: data.verification,
      };
      const returnedId = data.session_id || "";
      const nextMessages = [...messages, userEntry, assistantEntry];
      setMessages(nextMessages);
      if (resumedSessionId && returnedId && returnedId !== resumedSessionId) {
        // Backend đã quên session cũ nên tự tách id mới: mang lịch sử view
        // hiện tại sang id mới để cuộc chat tiếp diễn liền mạch.
        setMessageCache((cache) => {
          const { [cacheKey]: _dropped, ...rest } = cache;
          return { ...rest, [returnedId]: nextMessages };
        });
        setSessions((current) =>
          current.map((s) => (s.id === resumedSessionId ? { ...s, id: returnedId, stale: false } : s)),
        );
        setActiveSessionId(returnedId);
      } else {
        setMessageCache((cache) => ({ ...cache, [returnedId || cacheKey]: nextMessages }));
        if (returnedId) {
          setActiveSessionId(returnedId);
        }
      }
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
                      {document.stale ? " · cần upload lại" : ""}
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
                  <strong>
                    {session.title}
                    {session.stale ? " (local)" : ""}
                  </strong>
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

        {sessions.some((s) => s.stale) && (
          <p className="muted" style={{ padding: "8px 16px 0" }}>
            Backend đã reset — các mục (local) đang xem từ bản lưu trên trình duyệt. Gửi tin nhắn sẽ tự tạo đoạn chat
            mới.
          </p>
        )}

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
