import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './AppDemo.css';

type Metrics = {
  model?: string;
  episodes: number;
  deterministic?: boolean;
  success_rate: number; // 0..1
  avg_reward: number;
  collision_rate: number; // 0..1
  avg_steps: number;
  coordination_score: number;
  synchronization_score: number;
};

type Series = {
  episodes: number[];
  reward: number[];
  success_rate_cum: number[];
  collision_rate_cum: number[];
};

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function formatPct(x: number, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

function MetricCard(props: { label: string; value: string }) {
  return (
    <div className="card">
      <div className="cardLabel">{props.label}</div>
      <div className="cardValue">{props.value}</div>
    </div>
  );
}

export default function AppDemo() {
  const [episodes, setEpisodes] = useState(50);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('Ready.');
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [series, setSeries] = useState<Series | null>(null);

  const [demoRunning, setDemoRunning] = useState(false);
  const [demoGifUrl, setDemoGifUrl] = useState<string | null>(null);
  const [demoTrace, setDemoTrace] = useState<any | null>(null);

  const rewardData = useMemo(() => {
    if (!series) return [];
    return series.episodes.map((ep, i) => ({ ep, value: series.reward[i] ?? 0 }));
  }, [series]);

  const successData = useMemo(() => {
    if (!series) return [];
    return series.episodes.map((ep, i) => ({
      ep,
      value: series.success_rate_cum[i] ?? 0,
    }));
  }, [series]);

  const collisionData = useMemo(() => {
    if (!series) return [];
    return series.episodes.map((ep, i) => ({
      ep,
      value: series.collision_rate_cum[i] ?? 0,
    }));
  }, [series]);

  async function runEvaluation() {
    setLoading(true);
    setStatusText('Running evaluation…');
    setMetrics(null);
    setSeries(null);
    try {
      const res = await fetch(`${API_BASE}/api/run-eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes, deterministic: true }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { metrics: Metrics; series: Series };
      setMetrics(data.metrics);
      setSeries(data.series);
      setStatusText('Evaluation completed.');
    } catch (e: any) {
      setStatusText(`Failed: ${e?.message || 'unknown error'}`);
    } finally {
      setLoading(false);
    }
  }

  async function runDemoEpisode() {
    setDemoRunning(true);
    setStatusText('Running demo episode…');
    setDemoGifUrl(null);
    setDemoTrace(null);
    try {
      const res = await fetch(`${API_BASE}/api/run-demo-episode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: 60, deterministic: true }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { gifUrl: string; trace: any };
      setDemoGifUrl(data.gifUrl);
      setDemoTrace(data.trace);
      setStatusText('Demo episode ready.');
    } catch (e: any) {
      setStatusText(`Demo failed: ${e?.message || 'unknown error'}`);
    } finally {
      setDemoRunning(false);
    }
  }

  const headline = metrics ? `${metrics.model || 'PPO (CTDE)'} • ${metrics.episodes} episodes` : 'PPO (CTDE) • 50 episodes';
  const headlineSub = metrics
    ? `Success ${formatPct(metrics.success_rate)} • Avg Reward ${metrics.avg_reward.toFixed(1)}`
    : 'Click "Run Evaluation" to compute results.';

  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <div>
            <div className="title">HiveMind AI Demo</div>
            <div className="subtitle">{headline}</div>
            <div className="subtitle2">{headlineSub}</div>
          </div>

          <div className="controls">
            <div className="controlRow">
              <label className="controlLabel">Episodes</label>
              <select
                className="select"
                value={episodes}
                onChange={(e) => setEpisodes(parseInt(e.target.value, 10))}
                disabled={loading}
              >
                <option value={10}>10</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>

            <button className="btn" onClick={runEvaluation} disabled={loading}>
              {loading ? 'Running…' : 'Run Evaluation'}
            </button>

            <div className="status">{statusText}</div>
          </div>
        </div>

        <div className="heroBox" />

        <div className="gridCards">
          <MetricCard label="Success Rate" value={metrics ? formatPct(metrics.success_rate) : '—'} />
          <MetricCard label="Avg Reward" value={metrics ? metrics.avg_reward.toFixed(1) : '—'} />
          <MetricCard label="Collision Rate" value={metrics ? formatPct(metrics.collision_rate) : '—'} />
          <MetricCard label="Avg Steps" value={metrics ? metrics.avg_steps.toFixed(1) : '—'} />
          <MetricCard
            label="Coordination Score"
            value={metrics ? metrics.coordination_score.toFixed(3) : '—'}
          />
        </div>

        <div className="charts">
          <div className="chartCard">
            <div className="chartTitle">Reward vs Episodes</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={rewardData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ep" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#61dafb" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chartCard">
            <div className="chartTitle">Success Rate vs Episodes</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={successData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ep" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v: any) => formatPct(Number(v), 0)} />
                <Line type="monotone" dataKey="value" stroke="#22c55e" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chartCard">
            <div className="chartTitle">Collision Rate vs Episodes</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={collisionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ep" />
                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v: any) => formatPct(Number(v), 0)} />
                <Line type="monotone" dataKey="value" stroke="#ef4444" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="previewCard">
          <div className="chartTitle">Simulation Preview (Optional)</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10 }}>
            <button className="btn" onClick={runDemoEpisode} disabled={demoRunning}>
              {demoRunning ? 'Generating preview…' : 'Run Demo Episode'}
            </button>

            <div className="previewText">
              Evaluators see the simulation as a quick “input → action → next state” trace
              (and a motion GIF) directly in the dashboard—no CSV files.
            </div>

            <div style={{ border: '1px solid rgba(255,255,255,0.12)', borderRadius: 14, padding: 10 }}>
              {demoGifUrl ? (
                <img
                  src={demoGifUrl}
                  alt="Multi-agent motion preview"
                  style={{ width: '100%', borderRadius: 12, display: 'block' }}
                />
              ) : (
                <div style={{ color: 'rgba(229,231,235,0.8)', fontSize: 13, padding: 6 }}>
                  No demo generated yet. Click “Run Demo Episode”.
                </div>
              )}
            </div>

            <div className="ioTableWrap">
              <div className="chartTitle" style={{ marginTop: 2 }}>
                Use Case Scenario (Input → Action → Next)
              </div>

              <table className="ioTable">
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>Input</th>
                    <th>Action</th>
                    <th>Next</th>
                  </tr>
                </thead>
                <tbody>
                  {demoTrace?.trace?.slice(0, 8)?.map((row: any) => (
                    <tr key={row.t}>
                      <td>{row.t}</td>
                      <td>mean_dist={Number(row.input?.mean_distance).toFixed(3)}</td>
                      <td>
                        acc={Array.isArray(row.action)
                          ? row.action.flat().slice(0, 4).map((v: number) => Number(v).toFixed(2)).join(', ')
                          : '—'}
                      </td>
                      <td>mean_dist={Number(row.next?.mean_distance).toFixed(3)}</td>
                    </tr>
                  ))}
                  {(!demoTrace || !demoTrace.trace) && (
                    <tr>
                      <td colSpan={4} style={{ textAlign: 'center', color: 'rgba(229,231,235,0.8)' }}>
                        Run demo to show input/output steps.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

