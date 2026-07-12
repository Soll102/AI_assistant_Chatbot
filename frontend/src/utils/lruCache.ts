/**
 * LRU Cache for PDF documents.
 * Keeps only the N most recent PDFs in memory.
 * Automatically releases resources from evicted entries.
 */

export interface CacheEntry<T> {
  key: string;
  value: T;
  /** Callback invoked when entry is evicted */
  onEvict?: () => void;
}

export class LRUCache<T> {
  private capacity: number;
  private cache: Map<string, CacheEntry<T>>;

  constructor(capacity: number = 3) {
    this.capacity = capacity;
    this.cache = new Map();
  }

  get(key: string): T | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;

    // Move to most recently used (end of Map)
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T, onEvict?: () => void): void {
    // If key exists, delete old and re-insert
    if (this.cache.has(key)) {
      const old = this.cache.get(key)!;
      old.onEvict?.();
      this.cache.delete(key);
    }

    // Evict least recently used if at capacity
    while (this.cache.size >= this.capacity) {
      const lruKey = this.cache.keys().next().value;
      if (lruKey !== undefined) {
        const evicted = this.cache.get(lruKey)!;
        evicted.onEvict?.();
        this.cache.delete(lruKey);
      }
    }

    this.cache.set(key, { key, value, onEvict });
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (entry) {
      entry.onEvict?.();
    }
    return this.cache.delete(key);
  }

  clear(): void {
    for (const [_, entry] of this.cache) {
      entry.onEvict?.();
    }
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }

  keys(): string[] {
    return Array.from(this.cache.keys());
  }
}
