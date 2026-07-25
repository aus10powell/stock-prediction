import { afterEach, describe, expect, it, vi } from "vitest";
import { clientKey, createRateLimiter } from "./rateLimit";

afterEach(() => {
  vi.useRealTimers();
});

describe("createRateLimiter", () => {
  it("allows a burst up to capacity then rejects", () => {
    const consume = createRateLimiter(3, 1);

    expect(consume("a").allowed).toBe(true);
    expect(consume("a").allowed).toBe(true);
    expect(consume("a").allowed).toBe(true);

    const blocked = consume("a");
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
  });

  it("tracks callers independently", () => {
    const consume = createRateLimiter(1, 1);

    expect(consume("a").allowed).toBe(true);
    expect(consume("a").allowed).toBe(false);
    expect(consume("b").allowed).toBe(true);
  });

  it("refills over time", () => {
    vi.useFakeTimers();
    const consume = createRateLimiter(2, 1);

    consume("a");
    consume("a");
    expect(consume("a").allowed).toBe(false);

    vi.advanceTimersByTime(1_100);
    expect(consume("a").allowed).toBe(true);
  });
});

describe("clientKey", () => {
  it("prefers the first forwarded address", () => {
    const request = new Request("https://example.com", {
      headers: { "x-forwarded-for": "203.0.113.7, 70.41.3.18" },
    });
    expect(clientKey(request)).toBe("203.0.113.7");
  });

  it("falls back to the real IP header and then a placeholder", () => {
    expect(
      clientKey(
        new Request("https://example.com", {
          headers: { "x-real-ip": "198.51.100.4" },
        }),
      ),
    ).toBe("198.51.100.4");
    expect(clientKey(new Request("https://example.com"))).toBe("unknown");
  });
});
