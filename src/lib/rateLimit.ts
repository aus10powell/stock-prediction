type Bucket = {
  tokens: number;
  updatedAt: number;
};

export type RateLimitVerdict = {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
};

/**
 * Token bucket keyed by caller. Like the history cache this is per-instance, so
 * it protects the unauthenticated upstream data provider from a single busy
 * client rather than enforcing a global quota.
 */
export function createRateLimiter(capacity: number, refillPerSecond: number) {
  const buckets = new Map<string, Bucket>();

  return function consume(key: string): RateLimitVerdict {
    const now = Date.now();
    const bucket = buckets.get(key) ?? { tokens: capacity, updatedAt: now };

    const elapsedSeconds = (now - bucket.updatedAt) / 1000;
    bucket.tokens = Math.min(
      capacity,
      bucket.tokens + elapsedSeconds * refillPerSecond,
    );
    bucket.updatedAt = now;

    if (bucket.tokens < 1) {
      buckets.set(key, bucket);
      return {
        allowed: false,
        remaining: 0,
        retryAfterSeconds: Math.max(
          1,
          Math.ceil((1 - bucket.tokens) / refillPerSecond),
        ),
      };
    }

    bucket.tokens -= 1;
    buckets.set(key, bucket);

    if (buckets.size > 5_000) {
      for (const [candidate, entry] of buckets) {
        if (entry.tokens >= capacity && now - entry.updatedAt > 60_000) {
          buckets.delete(candidate);
        }
      }
    }

    return {
      allowed: true,
      remaining: Math.floor(bucket.tokens),
      retryAfterSeconds: 0,
    };
  };
}

export function clientKey(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) return forwarded.split(",")[0].trim();
  return request.headers.get("x-real-ip") ?? "unknown";
}
