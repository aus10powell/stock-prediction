type Entry<T> = {
  expiresAt: number;
  value: Promise<T>;
};

/**
 * Process-local TTL cache. Serverless instances are not shared, so this trims
 * repeated upstream calls within a warm instance rather than acting as a
 * cluster-wide cache. Responses also carry `s-maxage` for the CDN layer.
 */
export function createTtlCache<T>(ttlMs: number, maxEntries = 128) {
  const entries = new Map<string, Entry<T>>();

  return function withCache(key: string, load: () => Promise<T>): Promise<T> {
    const now = Date.now();
    const existing = entries.get(key);
    if (existing && existing.expiresAt > now) {
      return existing.value;
    }

    const value = load().catch((error) => {
      entries.delete(key);
      throw error;
    });
    entries.set(key, { expiresAt: now + ttlMs, value });

    if (entries.size > maxEntries) {
      for (const [candidate, entry] of entries) {
        if (entry.expiresAt <= now) entries.delete(candidate);
      }
      while (entries.size > maxEntries) {
        const oldest = entries.keys().next();
        if (oldest.done) break;
        entries.delete(oldest.value);
      }
    }

    return value;
  };
}
