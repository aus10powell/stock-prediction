"use client";

import type { CalibrationOutcome, CalibrationResult } from "@/lib/calibration";
import { percent } from "@/lib/format";

type Props = {
  result: CalibrationResult;
  ticker: string;
};

function verdict(outcome: CalibrationOutcome): string {
  const gap = outcome.meanPredicted - outcome.observedFrequency;
  if (outcome.skillScore <= 0) {
    return "No better than always predicting the historical base rate.";
  }
  if (Math.abs(gap) < 0.03) {
    return "Predicted and realized frequencies line up closely.";
  }
  return gap > 0
    ? "The model has been overconfident about this move happening."
    : "The model has been underconfident about this move happening.";
}

function OutcomeBlock({
  title,
  outcome,
}: {
  title: string;
  outcome: CalibrationOutcome;
}) {
  return (
    <div className="calibration-block">
      <h3>{title}</h3>
      <dl className="assumptions compact">
        <div>
          <dt>Mean predicted</dt>
          <dd>{percent(outcome.meanPredicted)}</dd>
        </div>
        <div>
          <dt>Actually happened</dt>
          <dd>{percent(outcome.observedFrequency)}</dd>
        </div>
        <div>
          <dt>Brier score</dt>
          <dd>{outcome.brierScore.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Skill vs base rate</dt>
          <dd>{percent(outcome.skillScore)}</dd>
        </div>
      </dl>
      <p className="callout">{verdict(outcome)}</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Predicted range</th>
              <th>Cases</th>
              <th>Mean predicted</th>
              <th>Actual</th>
            </tr>
          </thead>
          <tbody>
            {outcome.bins.map((bin) => (
              <tr key={bin.lowerBound}>
                <td>
                  {percent(bin.lowerBound)} – {percent(bin.upperBound)}
                </td>
                <td>{bin.count}</td>
                <td>{percent(bin.meanPredicted)}</td>
                <td>{percent(bin.observedFrequency)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function CalibrationPanel({ result, ticker }: Props) {
  return (
    <section className="panel reveal">
      <div className="panel-heading">
        <p className="eyebrow">Calibration backtest</p>
        <h2>Have these probabilities been trustworthy?</h2>
        <p>
          {result.samples} historical starting points for {ticker} between{" "}
          {result.firstOrigin} and {result.lastOrigin}, each simulated with only
          the data available at the time, then compared against what actually
          happened {result.horizonDays} trading day
          {result.horizonDays === 1 ? "" : "s"} later.
        </p>
      </div>

      <div className="calibration-grid">
        <OutcomeBlock
          title={`Up at least ${result.thresholdPercent}%`}
          outcome={result.up}
        />
        <OutcomeBlock
          title={`Down at least ${result.thresholdPercent}%`}
          outcome={result.down}
        />
      </div>

      <p className="disclaimer">
        Lower Brier scores are better, and a positive skill score means the
        simulation beat simply always quoting the historical base rate. Overlapping
        horizons make these samples correlated, so treat the figures as a sanity
        check rather than a precise measurement.
      </p>
    </section>
  );
}
