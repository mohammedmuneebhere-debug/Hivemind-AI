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
import './AppDashboard.css';

type Metrics = {
  model?: string;
  episodes: number;
  deterministic?: boolean;
  success_rate: number;
  avg_reward: number;
  collision_rate: number;
  avg_steps: number;
  coordination_score: number;
  synchronization_score?: number;
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

function formatApiError(prefix: string, e: unknown): string {
  if (e instanceof Error) return `${prefix}${e.message}`;
  return `${prefix}${String(e)}`;
}

async function readErrorMessage(res: Response): Promise<string> {
  const raw = await res.text();
  try {
    const j = JSON.parse(raw) as { detail?: unknown };
    const d = j?.detail;
    if (typeof d === 'string') return d;
    if (d != null && typeof d === 'object') {
      const o = d as Record<string, unknown>;
      if (typeof o.error === 'string') {
        const tail = typeof o.stderr === 'string' ? o.stderr.slice(-400) : '';
        return tail ? `${o.error}: ${tail}` : o.error;
      }
      return JSON.stringify(d);
    }
  } catch {
    /* ignore */
  }
  return raw.slice(0, 500) || `HTTP ${res.status}`;
}

function staticUrl(path: string) {
  if (!path) return '';
  return path.startsWith('http') ? path : `${API_BASE}${path}`;
}

function MetricCard(props: { label: string; value: string }) {
  return (
    <div className="dashCard">
      <div className="dashCardLabel">{props.label}</div>
      <div className="dashCardValue">{props.value}</div>
    </div>
  );
}

export default function AppDashboard() {
  const [episodes, setEpisodes] = useState(50);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('Ready.');
  const [statusError, setStatusError] = useState(false);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [series, setSeries] = useState<Series | null>(null);

  const [demoRunning, setDemoRunning] = useState(false);
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | null>(null);
  const [previewGifUrl, setPreviewGifUrl] = useState<string | null>(null);
  const [demoTrace, setDemoTrace] = useState<{ trace?: unknown[] } | null>(null);

  const [manualReady, setManualReady] = useState(false);
  const [manualBusy, setManualBusy] = useState(false);
  const [manualHint, setManualHint] = useState('Reset the environment, then execute steps with custom accelerations.');
  const [a1x, setA1x] = useState(0);
  const [a1y, setA1y] = useState(0);
  const [a2x, setA2x] = useState(0);
  const [a2y, setA2y] = useState(0);

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

  function setStatus(ok: string, err = false) {
    setStatusText(ok);
    setStatusError(err);
  }

  async function runEvaluation() {
    setLoading(true);
    setStatus('Running evaluation…', false);
    setMetrics(null);
    setSeries(null);
    try {
      const res = await fetch(`${API_BASE}/api/run-eval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episodes, deterministic: true }),
      });
      if (!res.ok) {
        throw new Error(await readErrorMessage(res));
      }
      const data = (await res.json()) as { metrics: Metrics; series: Series };
      setMetrics(data.metrics);
      setSeries(data.series);
      setStatus('Evaluation completed.', false);
    } catch (e: unknown) {
      setStatus(formatApiError('Error: ', e), true);
    } finally {
      setLoading(false);
    }
  }

  async function generateMp4Preview() {
    setDemoRunning(true);
    setStatus('Generating motion preview…', false);
    setPreviewVideoUrl(null);
    setPreviewGifUrl(null);
    setDemoTrace(null);
    try {
      const res = await fetch(`${API_BASE}/api/run-demo-episode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: 60, deterministic: true, regen: true }),
      });
      if (!res.ok) {
        throw new Error(await readErrorMessage(res));
      }
      const data = (await res.json()) as {
        gifUrl: string;
        videoUrl?: string;
        trace: { trace?: unknown[] };
      };
      setPreviewGifUrl(staticUrl(data.gifUrl));
      setPreviewVideoUrl(data.videoUrl ? staticUrl(data.videoUrl) : null);
      setDemoTrace(data.trace);
      setStatus('Preview ready.', false);
    } catch (e: unknown) {
      setStatus(formatApiError('Preview: ', e), true);
    } finally {
      setDemoRunning(false);
    }
  }

  async function manualReset() {
    setManualBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/manual-reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        throw new Error(await readErrorMessage(res));
      }
      setManualReady(true);
      const j = (await res.json()) as { mean_distance?: number };
      setManualHint(
        typeof j.mean_distance === 'number'
          ? `Environment reset. Mean distance to target: ${j.mean_distance.toFixed(3)}`
          : 'Environment reset.',
      );
    } catch (e: unknown) {
      setManualHint(formatApiError('Reset failed: ', e));
      setManualReady(false);
    } finally {
      setManualBusy(false);
    }
  }

  async function manualStep() {
    setManualBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/manual-step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agents: [
            [a1x, a1y],
            [a2x, a2y],
          ],
        }),
      });
      if (!res.ok) {
        throw new Error(await readErrorMessage(res));
      }
      const j = (await res.json()) as {
        reward?: number;
        mean_distance?: number;
        collision?: boolean;
        task_completed?: boolean;
        done?: boolean;
      };
      const parts = [
        `reward ${typeof j.reward === 'number' ? j.reward.toFixed(3) : '—'}`,
        `mean_dist ${typeof j.mean_distance === 'number' ? j.mean_distance.toFixed(3) : '—'}`,
        j.collision ? 'collision' : null,
        j.task_completed ? 'target reached' : null,
        j.done ? 'episode done' : null,
      ].filter(Boolean);
      setManualHint(parts.join(' · '));
    } catch (e: unknown) {
      setManualHint(formatApiError('Step failed: ', e));
    } finally {
      setManualBusy(false);
    }
  }

  const chartGrid = { stroke: 'rgba(255,255,255,0.06)' };
  const axisStyle = { fontSize: 11, fill: 'rgba(230,237,243,0.45)' };

  return (
    <div className="dashApp">
      <div className="dashContainer">
        <header className="dashHeader">
          <div className="dashBrand">
            <div className="dashLogo" aria-hidden>
              H
            </div>
            <div>
              <h1 className="dashTitle">HiveMind AI</h1>
              <p className="dashTagline">Multi-Agent CTDE Reinforcement Learning Dashboard</p>
            </div>
          </div>

          <div className="dashControls">
            <div className="dashControlsTop">
              <span className="dashEpisodesLabel">Episodes</span>
              <select
                className="dashSelect"
                value={episodes}
                onChange={(e) => setEpisodes(parseInt(e.target.value, 10))}
                disabled={loading}
                aria-label="Number of evaluation episodes"
              >
                <option value={10}>10</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
              <button type="button" className="dashBtnPrimary" onClick={runEvaluation} disabled={loading}>
                {loading ? 'Running…' : 'Run Evaluation'}
              </button>
            </div>
            <div className="dashStatusRow">
              <span className="dashStatusKey">Status:</span>
              <span className={statusError ? 'dashStatusVal dashStatusValErr' : 'dashStatusVal'}>{statusText}</span>
            </div>
          </div>
        </header>

        <section className="dashGridMetrics" aria-label="Aggregate metrics">
          <MetricCard label="Success rate" value={metrics ? formatPct(metrics.success_rate) : '—'} />
          <MetricCard label="Avg reward" value={metrics ? metrics.avg_reward.toFixed(1) : '—'} />
          <MetricCard label="Collision rate" value={metrics ? formatPct(metrics.collision_rate) : '—'} />
          <MetricCard label="Avg steps" value={metrics ? metrics.avg_steps.toFixed(1) : '—'} />
          <MetricCard label="Coordination" value={metrics ? metrics.coordination_score.toFixed(3) : '—'} />
        </section>

        <section className="dashCharts" aria-label="Performance charts">
          <div className="dashChartCard">
            <div className="dashChartTitle">Reward per episode</div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={rewardData}>
                <CartesianGrid {...chartGrid} strokeDasharray="3 3" />
                <XAxis dataKey="ep" tick={axisStyle} stroke={chartGrid.stroke} />
                <YAxis tick={axisStyle} stroke={chartGrid.stroke} />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                  labelStyle={{ color: '#e6edf3' }}
                />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="dashChartCard">
            <div className="dashChartTitle">Cumulative success rate</div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={successData}>
                <CartesianGrid {...chartGrid} strokeDasharray="3 3" />
                <XAxis dataKey="ep" tick={axisStyle} stroke={chartGrid.stroke} />
                <YAxis
                  domain={[0, 1]}
                  tick={axisStyle}
                  stroke={chartGrid.stroke}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={((value: number | string | undefined) => formatPct(Number(value ?? 0), 0)) as never}
                />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="dashChartCard">
            <div className="dashChartTitle">Cumulative collision rate</div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={collisionData}>
                <CartesianGrid {...chartGrid} strokeDasharray="3 3" />
                <XAxis dataKey="ep" tick={axisStyle} stroke={chartGrid.stroke} />
                <YAxis
                  domain={[0, 1]}
                  tick={axisStyle}
                  stroke={chartGrid.stroke}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                  formatter={((value: number | string | undefined) => formatPct(Number(value ?? 0), 0)) as never}
                />
                <Line type="monotone" dataKey="value" stroke="#f87171" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="dashBottom" aria-label="Simulation and manual control">
          <div className="dashPanel">
            <div className="dashPanelTitle">Simulation preview (MP4 / GIF)</div>
            <div className="dashVideoShell">
              {previewVideoUrl ? (
                <video key={previewVideoUrl} controls playsInline src={previewVideoUrl} aria-label="Multi-agent motion preview video" />
              ) : previewGifUrl ? (
                <img src={previewGifUrl} alt="Multi-agent motion preview" />
              ) : (
                <div className="dashPlaceholder">
                  <div className="dashPlayIcon" aria-hidden>
                    ▶
                  </div>
                  <p className="dashPlaceholderHint">
                    Click below to generate the motion preview. The player uses MP4 when the API serves a video asset; otherwise a GIF is shown.
                  </p>
                </div>
              )}
            </div>
            <button type="button" className="dashBtnGen" onClick={generateMp4Preview} disabled={demoRunning}>
              {demoRunning ? 'Generating…' : 'Generate MP4 preview'}
            </button>

            <div className="dashTrace">
              <div className="dashChartTitle">Use case trace (input → action → next)</div>
              <table className="dashIoTable">
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>Input</th>
                    <th>Action</th>
                    <th>Next</th>
                  </tr>
                </thead>
                <tbody>
                  {(demoTrace?.trace as Array<Record<string, unknown>> | undefined)?.slice(0, 8)?.map((row) => (
                    <tr key={String(row.t)}>
                      <td>{String(row.t)}</td>
                      <td>
                        mean_dist=
                        {Number((row.input as { mean_distance?: number } | undefined)?.mean_distance).toFixed(3)}
                      </td>
                      <td>
                        {Array.isArray(row.action)
                          ? (row.action as number[][])
                              .flat()
                              .slice(0, 4)
                              .map((v) => Number(v).toFixed(2))
                              .join(', ')
                          : '—'}
                      </td>
                      <td>
                        mean_dist=
                        {Number((row.next as { mean_distance?: number } | undefined)?.mean_distance).toFixed(3)}
                      </td>
                    </tr>
                  ))}
                  {(!demoTrace?.trace || demoTrace.trace.length === 0) && (
                    <tr>
                      <td colSpan={4} style={{ textAlign: 'center', color: 'rgba(230,237,243,0.55)' }}>
                        Generate a preview to populate the evaluator trace table.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="dashPanel">
            <div className="dashPanelTitle">Manual action control</div>
            <p className="dashManualHint" style={{ marginTop: 0 }}>
              Drive the mock multi-agent environment with explicit accelerations (normalized inputs in roughly [-1, 1]).
            </p>

            <div className="dashAgentBlock">
              <div className="dashAgentName">Agent 1</div>
              <div className="dashAccelRow">
                <div className="dashField">
                  <label htmlFor="a1x">Accel X</label>
                  <input id="a1x" type="number" step="0.05" value={a1x} onChange={(e) => setA1x(parseFloat(e.target.value) || 0)} />
                </div>
                <div className="dashField">
                  <label htmlFor="a1y">Accel Y</label>
                  <input id="a1y" type="number" step="0.05" value={a1y} onChange={(e) => setA1y(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
            </div>

            <div className="dashAgentBlock">
              <div className="dashAgentName">Agent 2</div>
              <div className="dashAccelRow">
                <div className="dashField">
                  <label htmlFor="a2x">Accel X</label>
                  <input id="a2x" type="number" step="0.05" value={a2x} onChange={(e) => setA2x(parseFloat(e.target.value) || 0)} />
                </div>
                <div className="dashField">
                  <label htmlFor="a2y">Accel Y</label>
                  <input id="a2y" type="number" step="0.05" value={a2y} onChange={(e) => setA2y(parseFloat(e.target.value) || 0)} />
                </div>
              </div>
            </div>

            <div className="dashManualFooter">
              <button type="button" className="dashBtnSecondary" onClick={manualReset} disabled={manualBusy}>
                Reset env
              </button>
              <button type="button" className="dashBtnExecute" onClick={manualStep} disabled={manualBusy || !manualReady}>
                Execute step
              </button>
            </div>
            <p className="dashManualHint">{manualHint}</p>
          </div>
        </section>
      </div>

      <footer className="dashSiteFooter">HiveMind AI — CTDE Multi-Agent RL</footer>
    </div>
  );
}
