interface ThumbnailEntry {
  dataUrl: string;
  width: number;
  height: number;
}

export class ThumbnailCache {
  private cache: Map<string, ThumbnailEntry>;
  private maxEntries: number;

  constructor(maxEntries: number = 200) {
    this.cache = new Map();
    this.maxEntries = maxEntries;
  }

  private key(pdfId: string, pageNum: number): string {
    return `${pdfId}:${pageNum}`;
  }

  get(pdfId: string, pageNum: number): ThumbnailEntry | undefined {
    const k = this.key(pdfId, pageNum);
    const entry = this.cache.get(k);
    if (entry) {
      this.cache.delete(k);
      this.cache.set(k, entry);
    }
    return entry;
  }

  set(pdfId: string, pageNum: number, entry: ThumbnailEntry): void {
    const k = this.key(pdfId, pageNum);
    if (this.cache.has(k)) this.cache.delete(k);
    while (this.cache.size >= this.maxEntries) {
      const lruKey = this.cache.keys().next().value;
      if (lruKey !== undefined) this.cache.delete(lruKey);
    }
    this.cache.set(k, entry);
  }

  has(pdfId: string, pageNum: number): boolean {
    return this.cache.has(this.key(pdfId, pageNum));
  }

  clearForPdf(pdfId: string): void {
    const prefix = `${pdfId}:`;
    for (const k of this.cache.keys()) {
      if (k.startsWith(prefix)) this.cache.delete(k);
    }
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}
