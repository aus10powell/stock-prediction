export type LeastSquaresFit = {
  /** Coefficients, one per design-matrix column. */
  beta: number[];
  /** (XᵀX)⁻¹, used for leverage and prediction intervals. */
  covarianceUnscaled: number[][];
  /** Residual standard error with degrees-of-freedom correction. */
  residualStd: number;
  residuals: number[];
  fitted: number[];
  rSquared: number;
  degreesOfFreedom: number;
};

function invert(matrix: number[][]): number[][] {
  const size = matrix.length;
  const work = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)),
  ]);

  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivotRow][column])) {
        pivotRow = row;
      }
    }

    const pivot = work[pivotRow][column];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error("Design matrix is singular; cannot fit the model.");
    }

    [work[column], work[pivotRow]] = [work[pivotRow], work[column]];

    for (let j = column; j < 2 * size; j++) {
      work[column][j] /= pivot;
    }

    for (let row = 0; row < size; row++) {
      if (row === column) continue;
      const factor = work[row][column];
      if (factor === 0) continue;
      for (let j = column; j < 2 * size; j++) {
        work[row][j] -= factor * work[column][j];
      }
    }
  }

  return work.map((row) => row.slice(size));
}

/**
 * Ordinary least squares via the normal equations. Fitting every column
 * simultaneously is what keeps correlated predictors (such as sine and cosine
 * terms at the same frequency) from biasing one another's coefficients.
 */
export function fitLeastSquares(
  design: number[][],
  target: number[],
): LeastSquaresFit {
  const rows = design.length;
  if (rows !== target.length) {
    throw new Error("Design matrix and target vector must have equal length.");
  }
  const columns = design[0]?.length ?? 0;
  if (columns === 0) {
    throw new Error("Design matrix must have at least one column.");
  }
  if (rows <= columns) {
    throw new Error("Not enough observations to fit the requested model.");
  }

  const normal = Array.from({ length: columns }, () =>
    new Array<number>(columns).fill(0),
  );
  const moment = new Array<number>(columns).fill(0);

  for (let r = 0; r < rows; r++) {
    const row = design[r];
    for (let i = 0; i < columns; i++) {
      moment[i] += row[i] * target[r];
      for (let j = i; j < columns; j++) {
        normal[i][j] += row[i] * row[j];
      }
    }
  }
  for (let i = 0; i < columns; i++) {
    for (let j = 0; j < i; j++) {
      normal[i][j] = normal[j][i];
    }
  }

  const covarianceUnscaled = invert(normal);
  const beta = covarianceUnscaled.map((row) =>
    row.reduce((sum, value, index) => sum + value * moment[index], 0),
  );

  const fitted = design.map((row) =>
    row.reduce((sum, value, index) => sum + value * beta[index], 0),
  );
  const residuals = target.map((value, index) => value - fitted[index]);

  const degreesOfFreedom = rows - columns;
  const residualSumOfSquares = residuals.reduce((sum, r) => sum + r * r, 0);
  const mean = target.reduce((sum, v) => sum + v, 0) / rows;
  const totalSumOfSquares = target.reduce((sum, v) => sum + (v - mean) ** 2, 0);

  return {
    beta,
    covarianceUnscaled,
    residualStd: Math.sqrt(residualSumOfSquares / degreesOfFreedom),
    residuals,
    fitted,
    rSquared:
      totalSumOfSquares === 0
        ? 0
        : 1 - residualSumOfSquares / totalSumOfSquares,
    degreesOfFreedom,
  };
}

/**
 * xᵀ(XᵀX)⁻¹x — how far a design row sits from the data used to fit the model.
 * Prediction intervals widen with this term, so extrapolation is reported as
 * less certain than in-sample fit.
 */
export function leverage(
  row: number[],
  covarianceUnscaled: number[][],
): number {
  let total = 0;
  for (let i = 0; i < row.length; i++) {
    let inner = 0;
    for (let j = 0; j < row.length; j++) {
      inner += covarianceUnscaled[i][j] * row[j];
    }
    total += row[i] * inner;
  }
  return total;
}
