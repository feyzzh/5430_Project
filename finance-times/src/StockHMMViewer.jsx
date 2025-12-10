// src/StockHMMViewer.jsx
import React, { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceArea,
} from "recharts";

const regimeColors = {
  bull: "#2ecc71",    // green
  bear: "#e74c3c",    // red
  rebound: "#3498db", // blue
};

export default function StockHMMViewer() {
  const [symbol, setSymbol] = useState("CRWD");
  const [period, setPeriod] = useState("10y");
  const [interval, setInterval] = useState("1d");
  const [K, setK] = useState(3);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const params = new URLSearchParams({
        symbol,
        period,
        interval,
        K: String(K),
      });

      const res = await fetch(`/api/hmm?${params.toString()}`);
      const data = await res.json();

      if (!res.ok) {
        setError(data.errors?.join("\n") || "Unknown error.");
        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch HMM data.");
    } finally {
      setLoading(false);
    }
  };

  // Transform time_series to something Recharts likes
  const chartData =
    result?.time_series?.map((d) => ({
      date: d.date ? d.date.slice(0, 10) : "",
      logReturn: d.logReturn,
      close: d.close,         // <-- closing price added
      regime: d.regime,
    })) || [];

  // Build contiguous regime segments for background shading
  const regimeSegments = useMemo(() => {
    const segments = [];
    if (!chartData || chartData.length === 0) return segments;

    let current = null;
    for (let i = 0; i < chartData.length; i++) {
      const point = chartData[i];
      const reg = point.regime;

      if (!reg) {
        // break any current segment
        if (current) {
          current.end = chartData[i - 1].date;
          segments.push(current);
          current = null;
        }
        continue;
      }

      if (!current || current.regime !== reg) {
        // close previous
        if (current) {
          current.end = chartData[i - 1].date;
          segments.push(current);
        }
        // start new
        current = { regime: reg, start: point.date, end: point.date };
      } else {
        // extend current
        current.end = point.date;
      }
    }

    if (current) {
      segments.push(current);
    }

    return segments;
  }, [chartData]);

  // We no longer need colored dots per-point, since we color the background.
  // const customDot = () => null;
  const customDot = (props) => { const { cx, cy, payload } = props; const color = regimeColors[payload.regime] || "#888888"; return <circle cx={cx} cy={cy} r={2} fill={color} />; };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "1.5rem" }}>
      <h2>HMM Regime Viewer</h2>

      <form
        onSubmit={handleSubmit}
        style={{
          marginBottom: "1rem",
          display: "flex",
          gap: "0.75rem",
          flexWrap: "wrap",
          alignItems: "flex-end",
        }}
      >
        <div>
          <label>
            Ticker:&nbsp;
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="CRWD"
            />
          </label>
        </div>

        <div>
          <label>
            Period:&nbsp;
            <input
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
              placeholder="10y"
            />
          </label>
        </div>

        <div>
          <label>
            Interval:&nbsp;
            <input
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              placeholder="1d"
            />
          </label>
        </div>

        <div>
          <label>
            K (states):&nbsp;
            <input
              type="number"
              min={2}
              max={5}
              value={K}
              onChange={(e) => setK(Number(e.target.value))}
              style={{ width: "4rem" }}
            />
          </label>
        </div>

        {/* Center the button on its own row */}
        <div
          style={{
            width: "100%",
            display: "flex",
            justifyContent: "center",
            marginTop: "0.5rem",
          }}
        >
          <button type="submit" disabled={loading}>
            {loading ? "Loading..." : "Run HMM"}
          </button>
        </div>
      </form>

      {error && (
        <div
          style={{
            color: "red",
            marginBottom: "0.5rem",
            whiteSpace: "pre-wrap",
          }}
        >
          {error}
        </div>
      )}

      {result && (
        <>
          <div style={{ marginBottom: "1rem" }}>
            <h3>
              {result.ticker} – {result.K} regimes
            </h3>
            <p>
              Log-likelihood (final):{" "}
              {result.log_likelihood_trace &&
              result.log_likelihood_trace.length
                ? result.log_likelihood_trace[
                    result.log_likelihood_trace.length - 1
                  ].toFixed(2)
                : "N/A"}
            </p>
            {result.regime_labels && (
              <p>Regime labels: {result.regime_labels.join(", ")}</p>
            )}
          </div>

          {/* Chart */}
          <div
            style={{
              background: "#fafafa",
              padding: "1rem",
              borderRadius: "8px",
            }}
          >
            <h4>Close Price & Log Returns with HMM Regimes</h4>
            <LineChart
              width={850}
              height={350}
              data={chartData}
              margin={{ top: 10, right: 40, bottom: 10, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              {/* Two Y-axes: left = logReturn, right = close price */}
              <YAxis
                yAxisId="left"
                tickFormatter={(v) => v?.toFixed?.(2)}
                label={{ value: "Log Return", angle: -90, position: "insideLeft" }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tickFormatter={(v) => v?.toFixed?.(2)}
                label={{ value: "Close", angle: 90, position: "insideRight" }}
              />
              <Tooltip />
              <Legend />

              {/* Background shading per regime */}
              {regimeSegments.map((seg, idx) => {
                const color = regimeColors[seg.regime] || "#cccccc";
                return (
                  <ReferenceArea
                    key={idx}
                    x1={seg.start}
                    x2={seg.end}
                    y1={Number.MIN_SAFE_INTEGER}
                    y2={Number.MAX_SAFE_INTEGER}
                    strokeOpacity={0}
                    fill={color}
                    fillOpacity={0.08}
                  />
                );
              })}

              {/* Log Returns line (left axis) */}
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="logReturn"
                stroke="#555555"
                dot={customDot}
                name="Log Return"
              />

              {/* Closing price line (right axis) */}
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="close"
                stroke="#000000"
                dot={false}
                name="Close"
              />
            </LineChart>

            <div style={{ marginTop: "0.5rem" }}>
              {Object.entries(regimeColors).map(([name, color]) => (
                <span key={name} style={{ marginRight: "1rem" }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 12,
                      height: 12,
                      borderRadius: "50%",
                      backgroundColor: color,
                      marginRight: 4,
                    }}
                  />
                  {name}
                </span>
              ))}
            </div>
          </div>

          {/* Regime summary */}
          {result.regime_summary && (
            <div style={{ marginTop: "1.5rem" }}>
              <h4>Regime Summary (mean features)</h4>
              <table
                style={{
                  borderCollapse: "collapse",
                  width: "100%",
                  fontSize: "0.9rem",
                }}
              >
                <thead>
                  <tr>
                    <th
                      style={{
                        borderBottom: "1px solid #ddd",
                        textAlign: "left",
                        padding: "0.25rem",
                      }}
                    >
                      Regime
                    </th>
                    <th
                      style={{
                        borderBottom: "1px solid #ddd",
                        textAlign: "right",
                        padding: "0.25rem",
                      }}
                    >
                      Log Returns
                    </th>
                    <th
                      style={{
                        borderBottom: "1px solid #ddd",
                        textAlign: "right",
                        padding: "0.25rem",
                      }}
                    >
                      Intraday Volatility
                    </th>
                    <th
                      style={{
                        borderBottom: "1px solid #ddd",
                        textAlign: "right",
                        padding: "0.25rem",
                      }}
                    >
                      Range
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.regime_summary).map(
                    ([regName, vals]) => (
                      <tr key={regName}>
                        <td
                          style={{
                            borderBottom: "1px solid #eee",
                            padding: "0.25rem",
                          }}
                        >
                          {regName}
                        </td>
                        <td
                          style={{
                            borderBottom: "1px solid #eee",
                            textAlign: "right",
                            padding: "0.25rem",
                          }}
                        >
                          {vals["Log Returns"]?.toFixed(4)}
                        </td>
                        <td
                          style={{
                            borderBottom: "1px solid #eee",
                            textAlign: "right",
                            padding: "0.25rem",
                          }}
                        >
                          {vals["Intraday Volatility"]?.toFixed(6)}
                        </td>
                        <td
                          style={{
                            borderBottom: "1px solid #eee",
                            textAlign: "right",
                            padding: "0.25rem",
                          }}
                        >
                          {vals["Range"]?.toFixed(4)}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
