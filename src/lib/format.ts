const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const signedPercentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
  signDisplay: "exceptZero",
});

export function percent(value: number): string {
  return percentFormatter.format(value);
}

export function signedPercent(value: number): string {
  return signedPercentFormatter.format(value);
}

export function money(value: number): string {
  return `$${value.toFixed(2)}`;
}
