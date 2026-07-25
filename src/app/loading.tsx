export default function Loading() {
  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden="true" />
      <header className="hero">
        <p className="byline">Austin Powell</p>
        <h1 className="brand">Stock Odds</h1>
        <p className="status">Loading market data and simulating paths…</p>
      </header>
    </div>
  );
}
