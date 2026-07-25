import { afterEach, describe, expect, it, vi } from "vitest";
import { createTtlCache } from "./cache";

afterEach(() => {
  vi.useRealTimers();
});

describe("createTtlCache", () => {
  it("serves repeated reads from one upstream call", async () => {
    const load = vi.fn(async () => "value");
    const withCache = createTtlCache<string>(1_000);

    const [first, second] = await Promise.all([
      withCache("key", load),
      withCache("key", load),
    ]);

    expect(first).toBe("value");
    expect(second).toBe("value");
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("reloads after the entry expires", async () => {
    vi.useFakeTimers();
    const load = vi.fn(async () => "value");
    const withCache = createTtlCache<string>(1_000);

    await withCache("key", load);
    vi.advanceTimersByTime(1_500);
    await withCache("key", load);

    expect(load).toHaveBeenCalledTimes(2);
  });

  it("does not cache failures", async () => {
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Error("upstream down"))
      .mockResolvedValueOnce("value");
    const withCache = createTtlCache<string>(10_000);

    await expect(withCache("key", load)).rejects.toThrow("upstream down");
    await expect(withCache("key", load)).resolves.toBe("value");
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("keeps distinct keys separate", async () => {
    const withCache = createTtlCache<string>(1_000);

    await expect(withCache("a", async () => "first")).resolves.toBe("first");
    await expect(withCache("b", async () => "second")).resolves.toBe("second");
  });
});
