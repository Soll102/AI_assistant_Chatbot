import { useRef, useCallback, useEffect } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { LRUCache } from '../utils/lruCache';
import { type PdfDocumentRef } from '../utils/pdfLoader';
import { ThumbnailCache } from '../utils/thumbnailCache';

try {
  pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url
  ).toString();
} catch {
  // fallback: let pdfjs use default worker
}

const API_BASE = `${import.meta.env.VITE_API_BASE ?? ''}/api/documents`;

export interface RenderPageOptions {
  pdfId: string;
  pageNum: number;
  canvas: HTMLCanvasElement;
  width?: number;
  scale?: number;
}

export interface RenderThumbnailOptions {
  pdfId: string;
  pageNum: number;
  canvas: HTMLCanvasElement;
  scale?: number;
}

export interface PdfPageRenderer {
  loadDocument: (pdfId: string, numPages?: number) => Promise<PdfDocumentRef>;
  renderPage: (options: RenderPageOptions) => Promise<void>;
  renderThumbnail: (options: RenderThumbnailOptions) => Promise<void>;
  getDocument: (pdfId: string) => pdfjsLib.PDFDocumentProxy | undefined;
  loadDocumentForThumbnails: (pdfId: string) => Promise<pdfjsLib.PDFDocumentProxy>;
  destroy: () => void;
}

interface CachedDocument {
  pdf: pdfjsLib.PDFDocumentProxy;
  pdfId: string;
  numPages: number;
}

const MAX_DOCS = 2;

export function usePdfViewer(): PdfPageRenderer {
  const docCacheRef = useRef<LRUCache<CachedDocument>>(
    new LRUCache<CachedDocument>(MAX_DOCS)
  );
  const thumbnailCacheRef = useRef<ThumbnailCache>(new ThumbnailCache(200));

  const loadDocument = useCallback(async (pdfId: string, numPages?: number): Promise<PdfDocumentRef> => {
    const cache = docCacheRef.current;
    const cached = cache.get(pdfId);
    if (cached) {
      return { pdfId, numPages: cached.numPages, filename: '' };
    }

    const pdfUrl = `${API_BASE}/${pdfId}/file`;

    const loadingTask = pdfjsLib.getDocument({
      url: pdfUrl,
      useSystemFonts: true,
      enableXfa: false,
    });

    const pdf = await loadingTask.promise;
    const totalPages = numPages ?? pdf.numPages;

    cache.set(pdfId, { pdf, pdfId, numPages: totalPages }, () => {
      pdf.destroy();
    });

    return { pdfId, numPages: totalPages, filename: '' };
  }, []);

  const renderPage = useCallback(async (options: RenderPageOptions): Promise<void> => {
    const { pdfId, pageNum, canvas, width, scale = 1 } = options;
    const cache = docCacheRef.current;
    const cached = cache.get(pdfId);

    if (!cached) {
      throw new Error(`PDF ${pdfId} not loaded. Call loadDocument() first.`);
    }

    const { pdf } = cached;
    const page = await pdf.getPage(pageNum);

    const dpr = window.devicePixelRatio || 1;
    const effectiveScale = scale * dpr;
    const viewport = page.getViewport({ scale: effectiveScale });

    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = width ? `${width}px` : `${viewport.width / dpr}px`;
    canvas.style.height = 'auto';

    const renderContext = {
      canvasContext: canvas.getContext('2d')!,
      viewport,
      intent: 'display' as const,
    };

    await page.render(renderContext).promise;
  }, []);

  const renderThumbnail = useCallback(async (options: RenderThumbnailOptions): Promise<void> => {
    const { pdfId, pageNum, canvas, scale = 0.3 } = options;
    const cache = docCacheRef.current;
    const cached = cache.get(pdfId);
    if (!cached) {
      throw new Error(`PDF ${pdfId} not loaded. Call loadDocument() first.`);
    }

    const { pdf } = cached;
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale });

    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const renderContext = {
      canvasContext: canvas.getContext('2d')!,
      viewport,
      intent: 'display' as const,
    };

    await page.render(renderContext).promise;

    try {
      const dataUrl = canvas.toDataURL('image/webp', 0.6);
      thumbnailCacheRef.current.set(pdfId, pageNum, {
        dataUrl,
        width: viewport.width,
        height: viewport.height,
      });
    } catch {
      // ignore caching failure
    }
  }, []);

  const getDocument = useCallback((pdfId: string): pdfjsLib.PDFDocumentProxy | undefined => {
    const cached = docCacheRef.current.get(pdfId);
    return cached?.pdf;
  }, []);

  const loadDocumentForThumbnails = useCallback(async (pdfId: string): Promise<pdfjsLib.PDFDocumentProxy> => {
    const cached = docCacheRef.current.get(pdfId);
    if (cached) return cached.pdf;

    const pdfUrl = `${API_BASE}/${pdfId}/file`;
    const loadingTask = pdfjsLib.getDocument({ url: pdfUrl, useSystemFonts: true, enableXfa: false });
    const pdf = await loadingTask.promise;

    docCacheRef.current.set(pdfId, { pdf, pdfId, numPages: pdf.numPages }, () => {
      pdf.destroy();
    });

    return pdf;
  }, []);

  const destroy = useCallback(() => {
    const cache = docCacheRef.current;
    thumbnailCacheRef.current.clear();
    cache.clear();
  }, []);

  useEffect(() => {
    return () => { docCacheRef.current.clear(); };
  }, []);

  return {
    loadDocument,
    renderPage,
    renderThumbnail,
    getDocument,
    loadDocumentForThumbnails,
    destroy,
  };
}
