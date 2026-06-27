import { useState, useEffect, useRef } from "react"
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement,
  ArcElement, Title, Tooltip, Legend, Filler
} from "chart.js"
import { Line, Doughnut, Bar } from "react-chartjs-2"

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend, Filler
)

// ── Constants ───────────────────────────────────────────────────────────
const WS_URL  = "ws://localhost:8000/ws"
const API_URL = "http://localhost:8000"

const RISK_COLORS = {
  LOW:      { bg: "#16a34a", text: "text-green-400",  border: "border-green-500" },
  MEDIUM:   { bg: "#ca8a04", text: "text-yellow-400", border: "border-yellow-500" },
  HIGH:     { bg: "#ea580c", text: "text-orange-400", border: "border-orange-500" },
  CRITICAL: { bg: "#dc2626", text: "text-red-400",    border: "border-red-500" },
}

// ── Hooks ───────────────────────────────────────────────────────────────
function useWebSocket() {
  const [telemetry, setTelemetry]   = useState(null)
  const [connected, setConnected]   = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws
      ws.onopen    = () => setConnected(true)
      ws.onclose   = () => { setConnected(false); setTimeout(connect, 2000) }
      ws.onerror   = () => ws.close()
      ws.onmessage = (e) => {
        try { setTelemetry(JSON.parse(e.data)) } catch {}
      }
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  return { telemetry, connected }
}

// ── Small components ────────────────────────────────────────────────────

function StatusDot({ connected }) {
  return (
    <span className={`inline-block w-2 h-2 rounded-full mr-2 ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`} />
  )
}

function MetricCard({ label, value, unit = "", color = "text-white" }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-semibold font-mono ${color}`}>
        {value ?? "—"}<span className="text-sm text-gray-500 ml-1">{unit}</span>
      </p>
    </div>
  )
}

function RiskBadge({ risk }) {
  const c = RISK_COLORS[risk] ?? RISK_COLORS.LOW
  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-bold border ${c.border} ${c.text}`}
      style={{ backgroundColor: c.bg + "22" }}
    >
      {risk}
    </span>
  )
}

function LaneStatusBadge({ status }) {
  const color = status === "Centered"
    ? "text-green-400 border-green-600"
    : "text-orange-400 border-orange-600"
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-bold border ${color}`}>
      {status}
    </span>
  )
}

// ── Charts ──────────────────────────────────────────────────────────────

function FPSChart({ history }) {
  const labels = history.map((_, i) => i)
  const data = {
    labels,
    datasets: [{
      label: "FPS",
      data: history,
      borderColor: "#4fc3f7",
      backgroundColor: "rgba(79,195,247,0.08)",
      fill: true,
      tension: 0.4,
      pointRadius: 0,
      borderWidth: 2,
    }]
  }
  const opts = {
    responsive: true,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: { min: 0, max: 60, ticks: { color: "#6b7280" }, grid: { color: "#1f2937" } }
    }
  }
  return <Line data={data} options={opts} />
}

function RiskDonut({ distribution }) {
  const labels = Object.keys(distribution)
  const values = Object.values(distribution)
  const colors = labels.map(l => RISK_COLORS[l]?.bg ?? "#6b7280")
  const data = {
    labels,
    datasets: [{ data: values, backgroundColor: colors, borderWidth: 0 }]
  }
  const opts = {
    responsive: true,
    animation: false,
    plugins: { legend: { position: "bottom", labels: { color: "#9ca3af", font: { size: 11 } } } }
  }
  return <Doughnut data={data} options={opts} />
}

function ClassCountBar({ counts }) {
  const labels = Object.keys(counts)
  const values = Object.values(counts)
  const data = {
    labels,
    datasets: [{
      label: "Count",
      data: values,
      backgroundColor: ["#4ade80","#f472b6","#fb923c","#818cf8","#34d399","#facc15"],
      borderRadius: 4,
    }]
  }
  const opts = {
    responsive: true,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#9ca3af" }, grid: { display: false } },
      y: { ticks: { color: "#9ca3af" }, grid: { color: "#1f2937" }, beginAtZero: true }
    }
  }
  return <Bar data={data} options={opts} />
}

// ── Main App ────────────────────────────────────────────────────────────

export default function App() {
  const { telemetry, connected } = useWebSocket()
  const [analytics, setAnalytics] = useState(null)
  const [running, setRunning]     = useState(false)
  const [source, setSource]       = useState("videos/dashcam2.mp4")
  const fpsHistory = useRef([])

  // Keep rolling FPS history
  useEffect(() => {
    if (telemetry?.fps) {
      fpsHistory.current.push(telemetry.fps)
      if (fpsHistory.current.length > 60) fpsHistory.current.shift()
    }
  }, [telemetry])

  // Poll analytics every 3s
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_URL}/api/analytics`)
        setAnalytics(await r.json())
      } catch {}
    }, 3000)
    return () => clearInterval(id)
  }, [])

  async function handleStart() {
    await fetch(`${API_URL}/api/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    })
    setRunning(true)
    fpsHistory.current = []
  }

  async function handleStop() {
    await fetch(`${API_URL}/api/stop`, { method: "POST" })
    setRunning(false)
  }

  const risk     = telemetry?.collision_risk ?? "LOW"
  const riskCol  = RISK_COLORS[risk] ?? RISK_COLORS.LOW
  const isCritical = risk === "CRITICAL" && telemetry?.warning_active

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* Critical warning banner */}
      {isCritical && (
        <div className="bg-red-600 text-white text-center py-2 text-sm font-bold animate-pulse">
          !! FORWARD COLLISION WARNING !!  &nbsp;|&nbsp;  {telemetry?.warning_msg}
        </div>
      )}

      {/* Top bar */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-cyan-400 tracking-wide">ADAS SIMULATOR</h1>
          <p className="text-xs text-gray-500">AI Driver Assistance System — Dashboard</p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-400">
            <StatusDot connected={connected} />
            {connected ? "Live" : "Disconnected"}
          </span>
          <RiskBadge risk={risk} />
        </div>
      </header>

      <main className="p-6 grid grid-cols-12 gap-4">

        {/* ── Control panel ── */}
        <section className="col-span-12 bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center gap-4 flex-wrap">
          <label className="text-xs text-gray-400">Video source</label>
          <input
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm flex-1 min-w-48 focus:outline-none focus:border-cyan-500"
            value={source}
            onChange={e => setSource(e.target.value)}
            disabled={running}
          />
          {!running ? (
            <button
              onClick={handleStart}
              className="bg-cyan-600 hover:bg-cyan-500 text-white px-5 py-1.5 rounded-lg text-sm font-medium transition"
            >
              ▶ Start
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="bg-red-700 hover:bg-red-600 text-white px-5 py-1.5 rounded-lg text-sm font-medium transition"
            >
              ■ Stop
            </button>
          )}
          {telemetry && (
            <span className="text-xs text-gray-500 ml-auto">
              Frame {telemetry.frame} · {telemetry.fps} FPS
            </span>
          )}
        </section>

        {/* ── Metric cards ── */}
        <div className="col-span-12 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <MetricCard label="Active Tracks"  value={telemetry?.active_tracks ?? 0} color="text-cyan-400" />
          <MetricCard label="FPS"            value={telemetry?.fps ?? 0}           color="text-green-400" />
          <MetricCard label="Lane Status"    value={telemetry?.lane_status ?? "—"} color={telemetry?.lane_status === "Centered" ? "text-green-400" : "text-orange-400"} />
          <MetricCard label="Lane Offset"    value={telemetry?.lane_offset ?? 0}   unit="cm" />
          <MetricCard label="TTC"            value={telemetry?.ttc ? telemetry.ttc.toFixed(1) : "—"} unit="s" color="text-yellow-400" />
          <MetricCard label="Total Vehicles" value={analytics?.total_vehicles ?? 0} color="text-purple-400" />
        </div>

        {/* ── FPS chart ── */}
        <section className="col-span-12 md:col-span-8 bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">FPS over time</p>
          <FPSChart history={[...fpsHistory.current]} />
        </section>

        {/* ── Risk donut ── */}
        <section className="col-span-12 md:col-span-4 bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Risk distribution</p>
          {analytics?.risk_distribution && Object.keys(analytics.risk_distribution).length > 0
            ? <RiskDonut distribution={analytics.risk_distribution} />
            : <p className="text-gray-600 text-sm text-center mt-8">No data yet</p>
          }
        </section>

        {/* ── Class counts ── */}
        <section className="col-span-12 md:col-span-6 bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Detected classes (current frame)</p>
          {telemetry?.class_counts && Object.keys(telemetry.class_counts).length > 0
            ? <ClassCountBar counts={telemetry.class_counts} />
            : <p className="text-gray-600 text-sm text-center mt-8">No detections</p>
          }
        </section>

        {/* ── Trip analytics ── */}
        <section className="col-span-12 md:col-span-6 bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Trip analytics</p>
          <div className="grid grid-cols-2 gap-3">
            {[
              ["Near Misses",       analytics?.near_misses      ?? 0, "text-red-400"],
              ["Lane Departures",   analytics?.lane_departures  ?? 0, "text-orange-400"],
              ["High Risk Events",  analytics?.high_risk_events ?? 0, "text-yellow-400"],
              ["Elapsed",           analytics ? `${analytics.elapsed_sec}s` : "—", "text-gray-300"],
            ].map(([label, val, col]) => (
              <div key={label} className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">{label}</p>
                <p className={`text-xl font-mono font-semibold ${col}`}>{val}</p>
              </div>
            ))}
          </div>
        </section>

      </main>
    </div>
  )
}
