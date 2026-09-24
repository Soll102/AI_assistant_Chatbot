import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bot, ChevronLeft, FileText, Loader2, MessageSquarePlus, Moon, Send, Sun, Trash2, Upload } from "lucide-react";
import katex from "katex";
import { marked } from "marked";
import "katex/dist/katex.min.css";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const MAX_QUESTION_LENGTH = 500;
// Khớp backend config.max_context_chunks mặc định (schemas.documents_skipped).
const MAX_CONTEXT_CHUNKS = 24;
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
      // null means "chưa chọn gì" -> mặc định là tất cả tài liệu.
      selectedDocumentIds: Array.isArray(parsed.selectedDocumentIds)
        ? parsed.selectedDocumentIds
        : null,
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
  // Phạm vi chat. null = chưa chọn -> hiểu là tất cả tài liệu khả dụng.
  const [selectedDocumentIds, setSelectedDocumentIds] = useState(
    initialCache?.selectedDocumentIds ?? null,
  );
  const [uploadNote, setUploadNote] = useState("");
  // PDFs sitting on the backend disk that the retrieval index does not know
  // about (e.g. uploaded before the index format changed). Surfaced as a
  // one-click repair instead of leaving the library silently incomplete.
  const [unindexedIds, setUnindexedIds] = useState([]);
  const [isReindexing, setIsReindexing] = useState(false);
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
    try {
      const saved = localStorage.getItem("rag-panel-sizes");
      // JSON corrupt trước đây throw trong render -> trắng app.
      return saved ? normalizePanelSizes(JSON.parse(saved)) : { sidebar: 250, preview: 560 };
    } catch {
      return { sidebar: 250, preview: 560 };
    }
  });
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("rag-theme");
    if (saved === "dark" || saved === "light") return saved;
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
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

  // Tài liệu backend còn giữ (bỏ qua bản local đã stale sau khi backend reset).
  const selectableDocuments = useMemo(() => documents.filter((item) => !item.stale), [documents]);

  // Phạm vi chat gửi lên backend. null/[] ở backend nghĩa là "tất cả", nên ở
  // đây phải luôn quy về danh sách id cụ thể để người dùng kiểm soát được.
  const effectiveSelection = useMemo(() => {
    const known = new Set(selectableDocuments.map((item) => item.id));
    if (selectedDocumentIds === null) return selectableDocuments.map((item) => item.id);
    return selectedDocumentIds.filter((id) => known.has(id));
  }, [selectedDocumentIds, selectableDocuments]);

  const selectionIsEmpty = selectableDocuments.length > 0 && effectiveSelection.length === 0;

  const localPdfUrl = activeDocumentId ? localPdfUrls[activeDocumentId] : "";
  const pdfUrl = localPdfUrl
    ? `${localPdfUrl}#page=${previewPage}&view=FitH`
    : activeDocumentId
      ? `${API_BASE}/api/documents/${activeDocumentId}/file#page=${previewPage}&view=FitH`
      : "";

  useEffect(() => {
    loadDocuments();
    loadSessions();
    loadIndexStatus();
  }, []);

  useEffect(() => {
    localStorage.setItem("rag-panel-sizes", JSON.stringify(panelSizes));
  }, [panelSizes]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("rag-theme", theme);
  }, [theme]);

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
          selectedDocumentIds,
        }),
      );
    } catch {
      // localStorage đầy hoặc bị chặn: bỏ qua, app vẫn chạy với state memory.
    }
  }, [sessions, messageCache, messages, documents, activeSessionId, activeDocumentId, selectedDocumentIds]);

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

  async function loadIndexStatus() {
    try {
      const response = await fetch(`${API_BASE}/api/documents/status`);
      if (!response.ok) return;
      const data = await response.json();
      setUnindexedIds(Array.isArray(data.unindexed) ? data.unindexed : []);
    } catch {
      // Backend không reachable: giữ nguyên trạng thái cũ.
    }
  }

  async function reindexDocuments() {
    setIsReindexing(true);
    try {
      const response = await fetch(`${API_BASE}/api/documents/reindex`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Index lại thất bại.");
      await loadDocuments();
      await loadIndexStatus();
      const lines = (data.indexed || []).map(
        (item) => `- **${item.filename}**: ${item.pages} trang, ${item.chunks} chunks`,
      );
      const errorLines = (data.errors || []).map((item) => `- ${item}`);
      const parts = [];
      if (lines.length) parts.push(`Đã index ${lines.length} tài liệu còn thiếu:\n${lines.join("\n")}`);
      else parts.push("Không còn tài liệu nào cần index.");
      if (errorLines.length) parts.push(`Không index được:\n${errorLines.join("\n")}`);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: parts.join("\n\n"), sources: [] },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Index lại lỗi: ${error.message}`, sources: [] },
      ]);
    } finally {
      setIsReindexing(false);
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
        // Backend ChatMessage không có sources (schema): giữ citations từ
        // cache local theo index khi nội dung trùng, thay vì vứt hết.
        const previous = messageCache[sessionId] ?? [];
        const mapped = data.length
          ? data.map((message, index) => {
              const kept = previous[index];
              const same =
                kept && kept.role === message.role && kept.content === message.content;
              return {
                role: message.role,
                content: message.content,
                sources: same && Array.isArray(kept.sources) ? kept.sources : [],
                documentsUsed: same && Array.isArray(kept.documentsUsed) ? kept.documentsUsed : [],
              };
            })
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
      // Backend unreachable cũng dọn local để user không kẹt entry chết.
      if (!response.ok && response.status !== 404) {
        // vẫn tiếp tục xoá local bên dưới
      }
    } catch {
      // vẫn tiếp tục xoá local bên dưới thay vì return sớm
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
      // Backend đã quên file (404 sau reset) hay unreachable thì vẫn xoá bản
      // lưu local để user không kẹt entry chết.
      if (!response.ok && response.status !== 404 && response.status !== 409) {
        // vẫn tiếp tục xoá local bên dưới
      }
      if (response.status === 409) {
        const data = await response.json().catch(() => ({}));
        setUploadNote(data.detail || "File đang bị khoá, đóng file và thử lại.");
      }
    } catch {
      // vẫn tiếp tục xoá local bên dưới
    }

    setLocalPdfUrls((current) => {
      if (current[documentId]) URL.revokeObjectURL(current[documentId].split("#")[0]);
      const { [documentId]: _removed, ...rest } = current;
      return rest;
    });

    // Không gọi setState trong updater (StrictMode double-invoke dễ sinh bug):
    // tính next từ state hiện tại rồi set từng phần riêng.
    const nextDocuments = documents.filter((document) => document.id !== documentId);
    setDocuments(nextDocuments);
    setSelectedDocumentIds((current) =>
      current === null ? null : current.filter((id) => id !== documentId),
    );
    if (activeDocumentId === documentId) {
      setActiveDocumentId(nextDocuments[0]?.id || "");
      setPreviewPage(1);
      setMessages([]);
    }
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

  async function uploadPdfs(event) {
    const files = Array.from(event.target.files || []);
    if (!files.length) return;
    // Guard client-side khớp backend (MAX_BATCH_FILES=20, MAX_UPLOAD_MB=50):
    // chặn sớm thay vì upload hàng trăm MB rồi mới 400/413.
    const MAX_BATCH_FILES = 20;
    const MAX_UPLOAD_MB = 50;
    if (files.length > MAX_BATCH_FILES) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Mỗi lần chỉ upload tối đa ${MAX_BATCH_FILES} file.`, sources: [] },
      ]);
      event.target.value = "";
      return;
    }
    const oversized = files.filter((file) => file.size > MAX_UPLOAD_MB * 1024 * 1024);
    if (oversized.length) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `File vượt quá ${MAX_UPLOAD_MB}MB: ${oversized.map((f) => f.name).join(", ")}.`,
          sources: [],
        },
      ]);
      event.target.value = "";
      return;
    }

    setIsUploading(true);
    setUploadNote(files.length > 1 ? `Đang index ${files.length} file...` : "Đang index...");
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    try {
      const response = await fetch(`${API_BASE}/api/documents/batch`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Upload thất bại.");
      }
      const summaries = Array.isArray(data) ? data : [data];
      const batchErrors = response.headers.get("X-Batch-Errors") || "";

      // Giữ blob URL để preview local, không phụ thuộc file trên backend.
      // Ghép theo tên file vì backend bỏ qua file lỗi nên thứ tự có thể lệch.
      const filesByName = new Map();
      for (const file of files) {
        if (!filesByName.has(file.name)) filesByName.set(file.name, file);
      }
      setLocalPdfUrls((current) => {
        const next = { ...current };
        for (const summary of summaries) {
          const file = filesByName.get(summary.filename);
          if (file && !next[summary.id]) next[summary.id] = URL.createObjectURL(file);
        }
        return next;
      });

      setDocuments((current) => [
        ...summaries,
        ...current.filter((item) => !summaries.some((summary) => summary.id === item.id)),
      ]);
      setSelectedDocumentIds((current) =>
        current === null
          ? null
          : [...new Set([...current, ...summaries.map((summary) => summary.id)])],
      );
      setActiveDocumentId(summaries[0]?.id ?? "");
      setPreviewPage(1);

      const failed = files.length - summaries.length;
      const lines = summaries.map(
        (summary) => `- **${summary.filename}**: ${summary.pages} trang, ${summary.chunks} chunks`,
      );
      const warning =
        failed > 0
          ? `\n\n${failed} file không index được.${batchErrors ? ` Chi tiết: ${batchErrors}` : " (xem console để biết chi tiết)."}` 
          : "";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Đã index ${summaries.length}/${files.length} file:\n${lines.join("\n")}${warning}`,
          sources: [],
        },
      ]);
      if (failed > 0) {
        setUploadNote(`${summaries.length}/${files.length} file được index.`);
      }
      await loadSessions();
    } catch (error) {
      setUploadNote("");
      setMessages((current) => [
        ...current,
        { role: "assistant", content: `Upload lỗi: ${error.message}`, sources: [] },
      ]);
    } finally {
      setIsUploading(false);
      event.target.value = "";
      window.setTimeout(() => setUploadNote(""), 4000);
    }
  }

  function toggleDocumentSelection(documentId) {
    setSelectedDocumentIds((current) => {
      const base = current === null ? selectableDocuments.map((item) => item.id) : current;
      return base.includes(documentId)
        ? base.filter((id) => id !== documentId)
        : [...base, documentId];
    });
  }

  function selectAllDocuments() {
    setSelectedDocumentIds(selectableDocuments.map((item) => item.id));
  }

  function clearDocumentSelection() {
    setSelectedDocumentIds([]);
  }

  async function askQuestion(event) {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || isAsking) return;
    if (selectionIsEmpty) return;

    const resumedSessionId = activeSessionId;
    const cacheKey = resumedSessionId || "__none";
    // Tài liệu stale (backend đã quên sau reset) đã bị loại khỏi effectiveSelection.
    // Có tài liệu nhưng chưa chọn gì thì chặn ở trên, không gửi mảng rỗng —
    // backend hiểu [] là "tất cả".
    const scopeIds = selectableDocuments.length ? effectiveSelection : null;
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
          document_id: null,
          document_ids: scopeIds,
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
        documentsUsed: data.documents_used || [],
        documentsSkipped: data.documents_skipped || [],
        queryRewritten: Boolean(data.query_rewritten),
      };
      const returnedId = data.session_id || "";
      // Dùng functional update để tránh stale closure khi openSession/reindex
      // xen vào giữa await (trước đây [...messages, ...] mất tin).
      let nextMessages = [];
      setMessages((current) => {
        nextMessages = [...current, userEntry, assistantEntry];
        return nextMessages;
      });
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
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title={theme === "dark" ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
          >
            {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>

        <button className="primary-button" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
          {isUploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
          {isUploading ? uploadNote || "Đang index..." : "Upload PDF"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={uploadPdfs}
        />
        {uploadNote && !isUploading && <p className="muted upload-note">{uploadNote}</p>}

        <section className="document-list">
          <div className="document-list-header">
            <h2>Tài liệu</h2>
            {selectableDocuments.length > 0 && (
              <span className="selection-count">
                {effectiveSelection.length}/{selectableDocuments.length} trong phạm vi
              </span>
            )}
          </div>
          {selectableDocuments.length > 1 && (
            <div className="selection-actions">
              <button type="button" onClick={selectAllDocuments}>Chọn tất cả</button>
              <button type="button" onClick={clearDocumentSelection}>Bỏ chọn</button>
            </div>
          )}
          {unindexedIds.length > 0 && (
            <div className="index-warning">
              <p>
                {unindexedIds.length} PDF có trên đĩa nhưng chưa được index — chúng sẽ không
                xuất hiện trong câu trả lời.
              </p>
              <button type="button" onClick={reindexDocuments} disabled={isReindexing}>
                {isReindexing ? "Đang index..." : `Index ${unindexedIds.length} file còn thiếu`}
              </button>
            </div>
          )}
          {documents.length === 0 ? (
            <p className="muted">Chưa có PDF nào.</p>
          ) : (
            documents.map((document) => {
              const selected = !document.stale && effectiveSelection.includes(document.id);
              return (
                <div className="document-row" key={document.id}>
                  <label className="document-check" title="Đưa tài liệu này vào phạm vi trả lời">
                    <input
                      type="checkbox"
                      checked={selected}
                      disabled={document.stale}
                      onChange={() => toggleDocumentSelection(document.id)}
                    />
                  </label>
                  <button
                    className={`document-item ${document.id === activeDocumentId ? "active" : ""} ${
                      selected ? "in-scope" : ""
                    }`}
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
              );
            })
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
                  {message.toolName && <span>Cách tìm: {formatToolName(message.toolName)}</span>}
                  {message.verification && (
                    <span>Đối chiếu: {formatVerification(message.verification)}</span>
                  )}
                  {message.documentsUsed?.length > 0 && (
                    <span>Dùng {message.documentsUsed.length} tài liệu</span>
                  )}
                </div>
              )}
              {message.queryRewritten && (
                <p className="message-warning">
                  Câu hỏi nối tiếp không khớp từ khoá nào nên đã được mở rộng bằng lượt hỏi trước.
                </p>
              )}
              {message.documentsSkipped?.length > 0 && (
                <p className="message-warning">
                  {message.documentsSkipped.length} tài liệu không lọt vào ngữ cảnh do giới hạn
                  ngân sách (tối đa {MAX_CONTEXT_CHUNKS} đoạn).
                  {formatSkippedDocs(message.documentsSkipped, documents)} Câu trả lời có thể chưa
                  bao quát hết.
                </p>
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
            placeholder={
              selectableDocuments.length === 0
                ? "Upload hoặc chọn PDF trước..."
                : selectionIsEmpty
                  ? "Chọn ít nhất một tài liệu để hỏi..."
                  : effectiveSelection.length === 1
                    ? "Hỏi về tài liệu đang chọn..."
                    : `Hỏi trên ${effectiveSelection.length} tài liệu...`
            }
            maxLength={MAX_QUESTION_LENGTH}
            rows={1}
          />
          <span className={`char-counter ${question.length >= MAX_QUESTION_LENGTH ? "limit" : ""}`}>
            {question.length}/{MAX_QUESTION_LENGTH}
          </span>
          <button type="submit" disabled={!question.trim() || isAsking || selectionIsEmpty}>
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
  const fallbackWidth = typeof window !== "undefined" ? window.innerWidth : 1280;
  const sizesObj = sizes && typeof sizes === "object" ? sizes : {};
  return {
    sidebar: clamp(Number(sizesObj.sidebar) || 250, 200, 420),
    preview: clamp(Number(sizesObj.preview) || 560, 260, Math.max(360, fallbackWidth - 540)),
  };
}

function formatToolName(name) {
  const labels = {
    search_pdf: "tìm trong tài liệu",
    summarize_pdf: "tóm tắt",
    compare_pdfs: "so sánh nhiều tài liệu",
    list_pdfs: "liệt kê tài liệu",
  };
  return labels[name] || name;
}

function formatVerification(value) {
  if (!value) return "";
  if (value.startsWith("supported")) return "câu trả lời khớp tài liệu";
  if (value.startsWith("revised")) return "đã sửa cho khớp tài liệu";
  if (value.startsWith("no_evidence")) return "tài liệu không đủ thông tin";
  if (value === "skipped" || value === "disabled" || value === "unverified")
    return "chưa đối chiếu";
  return value;
}

function formatSkippedDocs(ids, documents) {
  if (!Array.isArray(ids) || !ids.length) return "";
  const names = ids.map((id) => {
    const found = (documents || []).find((doc) => doc.id === id);
    return found ? found.filename : id.slice(0, 8);
  });
  return ` (${names.join(", ")})`;
}

function sanitizeRenderedHtml(html) {
  // marked không sanitize: tên file/answer LLM chứa <script>/<iframe> hoặc
  // on*=... sẽ thành Stored XSS. Lọc tối thiểu mà không cần thêm dep.
  return String(html || "")
    .replace(/<script[\s\S]*?<\/script\s*>/gi, "")
    .replace(/<(iframe|object|embed|form|input|button|select|textarea|link|meta|base|style)[\s\S]*?(?:<\/\1\s*>|\/>|>)/gi, "")
    .replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(/(href|src|xlink:href)\s*=\s*("|')\s*javascript:[^"']*(")/gi, '$1=$2#$3');
}

function renderMarkdown(content) {
  return sanitizeRenderedHtml(marked.parse(formatMath(content || "")));
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
  // Backend đã giới hạn số nguồn (1-2 cho câu hỏi thường, tối đa 4 cho so
  // sánh), nên hiển thị hết thay vì tự cắt thêm.
  return (
    <div className="sources">
      <strong>Đoạn liên quan</strong>
      {sources.map((source, index) => (
        <details key={`${source.document_id}-${source.preview_page || source.page}-${index}`}>
          <summary>
            <span>
              Đoạn {index + 1}
              <small className="source-origin">
                {" "}
                {source.filename} · trang {source.preview_page || source.page}
              </small>
            </span>
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
