import React, { useRef, useEffect, useState, useCallback } from 'react';

interface ThumbnailStripProps {
  pdfId: string | null;
  currentPage: number;
  totalPages: number;
  onPageClick: (page: number) => void;
  onRenderThumbnail?: (pageNum: number, canvas: HTMLCanvasElement) => Promise<void>;
  getThumbnailCache?: (pageNum: number) => { dataUrl: string; width: number; height: number } | undefined;
}

export const ThumbnailStrip: React.FC<ThumbnailStripProps> = ({
  pdfId,
  currentPage,
  totalPages,
  onPageClick,
  onRenderThumbnail,
  getThumbnailCache,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleRange, setVisibleRange] = useState<{ start: number; end: number }>({ start: 1, end: 30 });

  const updateVisibleRange = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const scrollTop = container.scrollTop;
    const height = container.clientHeight;
    const itemHeight = 104;
    const buffer = 4;
    const start = Math.max(1, Math.floor(scrollTop / itemHeight) - buffer);
    const end = Math.min(totalPages, Math.ceil((scrollTop + height) / itemHeight) + buffer);
    setVisibleRange({ start, end });
  }, [totalPages]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    updateVisibleRange();
    container.addEventListener('scroll', updateVisibleRange, { passive: true });
    return () => container.removeEventListener('scroll', updateVisibleRange);
  }, [updateVisibleRange]);

  useEffect(() => {
    updateVisibleRange();
  }, [totalPages, pdfId, updateVisibleRange]);

  if (totalPages === 0 || !pdfId) {
    return (
      <div className="w-16 flex items-center justify-center bg-gray-50 border-r border-gray-200 h-full">
        <div className="animate-pulse w-4 h-4 bg-gray-300 rounded-full" />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-16 overflow-y-auto bg-gray-50 border-r border-gray-200"
      style={{ maxHeight: '100%' }}
    >
      <div className="py-2 flex flex-col items-center gap-1">
        {Array.from({ length: totalPages }, (_, i) => i + 1).map((pageNum) => (
          <ThumbnailItem
            key={pageNum}
            pdfId={pdfId}
            pageNum={pageNum}
            isActive={pageNum === currentPage}
            isVisible={pageNum >= visibleRange.start && pageNum <= visibleRange.end}
            onClick={() => onPageClick(pageNum)}
            onRenderThumbnail={onRenderThumbnail}
            cachedEntry={getThumbnailCache?.(pageNum)}
          />
        ))}
      </div>
    </div>
  );
};

interface ThumbnailItemProps {
  pdfId: string;
  pageNum: number;
  isActive: boolean;
  isVisible: boolean;
  onClick: () => void;
  onRenderThumbnail?: (pageNum: number, canvas: HTMLCanvasElement) => Promise<void>;
  cachedEntry?: { dataUrl: string; width: number; height: number };
}

const ThumbnailItem: React.FC<ThumbnailItemProps> = ({
  pdfId,
  pageNum,
  isActive,
  isVisible,
  onClick,
  onRenderThumbnail,
  cachedEntry,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderedRef = useRef(false);

  useEffect(() => {
    if (!isVisible || !onRenderThumbnail || renderedRef.current) return;
    renderedRef.current = true;
    const canvas = canvasRef.current;
    if (!canvas) return;

    onRenderThumbnail(pageNum, canvas).catch(() => {});
  }, [pdfId, pageNum, isVisible, onRenderThumbnail]);

  useEffect(() => {
    renderedRef.current = false;
  }, [pdfId]);

  const showCached = cachedEntry && !renderedRef.current;

  return (
    <button
      onClick={onClick}
      className={`
        w-12 h-16 flex-shrink-0 border-2 rounded overflow-hidden
        transition-all duration-150 flex items-center justify-center
        ${isActive
          ? 'border-blue-500 shadow-md'
          : 'border-transparent hover:border-gray-300 opacity-70 hover:opacity-100'
        }
        ${!isVisible ? 'invisible' : ''}
      `}
      title={`Trang ${pageNum}`}
    >
      {showCached ? (
        <img
          src={cachedEntry.dataUrl}
          alt={`Page ${pageNum}`}
          className="w-full h-full object-cover"
        />
      ) : (
        <canvas
          ref={canvasRef}
          className="w-full h-full object-cover"
        />
      )}
    </button>
  );
};

export default ThumbnailStrip;
