import { useRef, useEffect, useState, useCallback } from 'react';
import { usePdfViewer, type PdfPageRenderer } from '../hooks/usePdfViewer';

interface PdfViewerProps {
  pdfId: string | null;
  pageNum?: number;
  onPageChange?: (page: number) => void;
  onTotalPagesChange?: (pages: number) => void;
  initialPage?: number;
  viewer?: PdfPageRenderer;
}

const CANVAS_PREV = 0;
const CANVAS_CURRENT = 1;
const CANVAS_NEXT = 2;

export const PdfViewer: React.FC<PdfViewerProps> = ({
  pdfId,
  pageNum,
  onPageChange,
  onTotalPagesChange,
  initialPage = 1,
  viewer: externalViewer,
}) => {
  const internalViewer = usePdfViewer();
  const viewer = externalViewer || internalViewer;

  const [currentPage, setCurrentPage] = useState(initialPage);
  const [totalPages, setTotalPages] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canvasRefs = [
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
  ];

  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(600);
  const renderIdRef = useRef(0);
  const currentPageRef = useRef(currentPage);
  const pdfIdRef = useRef(pdfId);

  currentPageRef.current = currentPage;
  pdfIdRef.current = pdfId;

  const getCanvas = useCallback((idx: number): HTMLCanvasElement | null => {
    return canvasRefs[idx].current;
  }, []);

  const renderPageToCanvas = useCallback(async (
    pageNum: number,
    canvasIdx: number,
  ) => {
    const canvas = getCanvas(canvasIdx);
    if (!canvas || !pdfId) return;

    try {
      await viewer.renderPage({
        pdfId,
        pageNum,
        canvas,
        width: containerWidth,
        scale: 1,
      });
    } catch (err) {
      // ignore stale renders
    }
  }, [pdfId, containerWidth, viewer, getCanvas]);

  const scheduleAdjacentRender = useCallback((page: number, canvasIdx: number) => {
    const id = ++renderIdRef.current;
    setTimeout(async () => {
      if (id !== renderIdRef.current) return;
      if (pdfIdRef.current !== pdfId) return;
      if (page < 1 || page > totalPages) return;
      await renderPageToCanvas(page, canvasIdx);
    }, 50);
  }, [pdfId, totalPages, renderPageToCanvas]);

  useEffect(() => {
    if (!pdfId) return;

    let cancelled = false;
    setLoading(true);
    setError(null);
    setCurrentPage(pageNum ?? 1);

    viewer.loadDocument(pdfId)
      .then((info) => {
        if (cancelled) return;
        setTotalPages(info.numPages);
        onTotalPagesChange?.(info.numPages);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load PDF');
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, [pdfId, viewer, onTotalPagesChange]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const renderAll = useCallback(async (page: number) => {
    if (!pdfId) return;
    const id = ++renderIdRef.current;

    await renderPageToCanvas(page, CANVAS_CURRENT);

    if (id !== renderIdRef.current) return;
    if (page > 1) {
      scheduleAdjacentRender(page - 1, CANVAS_PREV);
    }
    if (id !== renderIdRef.current) return;
    if (page < totalPages) {
      scheduleAdjacentRender(page + 1, CANVAS_NEXT);
    }
  }, [pdfId, totalPages, renderPageToCanvas, scheduleAdjacentRender]);

  useEffect(() => {
    renderAll(currentPage);
  }, [currentPage, renderAll]);

  useEffect(() => {
    if (!pageNum || !pdfId || loading || totalPages === 0) return;
    if (pageNum >= 1 && pageNum <= totalPages && pageNum !== currentPage) {
      setCurrentPage(pageNum);
    }
  }, [pageNum, pdfId]);

  const goToPage = useCallback((page: number) => {
    if (page < 1 || page > totalPages) return;
    setCurrentPage(page);
    onPageChange?.(page);
  }, [totalPages, onPageChange]);

  const goToPrev = useCallback(() => goToPage(currentPage - 1), [currentPage, goToPage]);
  const goToNext = useCallback(() => goToPage(currentPage + 1), [currentPage, totalPages, goToPage]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!pdfId) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        goToNext();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        goToPrev();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [pdfId, goToNext, goToPrev]);

  if (!pdfId) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-100 text-gray-500 text-sm">
        <div className="text-center p-8">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          <p className="mt-3 text-gray-500">Chưa chọn tài liệu PDF</p>
          <p className="text-xs text-gray-400 mt-1">Upload hoặc chọn một PDF từ danh sách</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-100">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto" />
          <p className="mt-3 text-sm text-gray-500">Đang tải PDF...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-100">
        <div className="text-center text-red-500">
          <p className="text-sm">Lỗi: {error}</p>
          <button
            onClick={() => viewer.loadDocument(pdfId)}
            className="mt-2 text-xs text-blue-500 underline"
          >
            Thử lại
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-100">
      <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center gap-1">
          <button
            onClick={goToPrev}
            disabled={currentPage <= 1}
            className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title="Trang trước (←)"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <span className="text-sm font-medium text-gray-700 mx-2">
            Trang{' '}
            <input
              type="number"
              value={currentPage}
              min={1}
              max={totalPages}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                if (val >= 1 && val <= totalPages) goToPage(val);
              }}
              className="w-12 text-center border rounded px-1 py-0.5 text-sm"
            />{' '}
            / {totalPages}
          </span>
          <button
            onClick={goToNext}
            disabled={currentPage >= totalPages}
            className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            title="Trang sau (→)"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="flex-1 overflow-auto bg-gray-200"
      >
        <div className="relative flex justify-center py-4 min-h-full">
          <div className="relative bg-white shadow-lg rounded-sm overflow-hidden" style={{ width: containerWidth }}>
            {[CANVAS_PREV, CANVAS_CURRENT, CANVAS_NEXT].map((idx) => (
              <canvas
                key={idx}
                ref={canvasRefs[idx]}
                className="block transition-opacity duration-200"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: 'auto',
                  opacity: idx === CANVAS_CURRENT ? 1 : 0,
                  pointerEvents: idx === CANVAS_CURRENT ? 'auto' : 'none',
                  zIndex: idx === CANVAS_CURRENT ? 2 : 1,
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PdfViewer;
