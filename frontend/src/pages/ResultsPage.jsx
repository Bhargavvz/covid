import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getResult, getResultSummary, getHeatmapUrl, getSlices } from '../api'

const SEVERITY_COLORS = {
    Normal: 'var(--severity-normal)',
    Mild: 'var(--severity-mild)',
    Moderate: 'var(--severity-moderate)',
    Severe: 'var(--severity-severe)',
}

export default function ResultsPage() {
    const { scanId } = useParams()
    const [result, setResult] = useState(null)
    const [summary, setSummary] = useState(null)
    const [slices, setSlices] = useState([])
    const [activeSlice, setActiveSlice] = useState(0)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        loadResults()
    }, [scanId])

    const loadResults = async () => {
        try {
            setLoading(true)
            const [resultData, summaryData] = await Promise.all([
                getResult(scanId).catch(() => null),
                getResultSummary(scanId).catch(() => null),
            ])
            setResult(resultData)
            setSummary(summaryData)

            try {
                const sliceData = await getSlices(scanId)
                setSlices(sliceData.slices || [])
            } catch {
                setSlices([])
            }
        } catch (err) {
            setError(err.message || 'Failed to load results')
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Loading analysis results...</p>
            </div>
        )
    }

    if (error || !result) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">⚠️</div>
                <div className="empty-state-title">Results Not Found</div>
                <p>{error || 'No analysis results available for this scan.'}</p>
                <Link to="/" className="btn btn-primary" style={{ marginTop: '1rem' }}>
                    Upload New Scan
                </Link>
            </div>
        )
    }

    const severityColor = SEVERITY_COLORS[result.severity_label] || 'var(--text-primary)'
    const damageColor = result.damage_percent > 50 ? 'var(--severity-severe)' :
        result.damage_percent > 25 ? 'var(--severity-moderate)' :
            result.damage_percent > 10 ? 'var(--severity-mild)' : 'var(--severity-normal)'

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">Analysis Results</h1>
                <p className="page-subtitle">Scan ID: {scanId}</p>
            </div>

            {/* Stats grid */}
            <div className="stats-grid">
                <div className="stat-card">
                    <div className="stat-icon severity">🔬</div>
                    <div>
                        <div className="stat-value" style={{ color: severityColor }}>
                            {result.severity_label}
                        </div>
                        <div className="stat-label">Severity</div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon damage">🫁</div>
                    <div>
                        <div className="stat-value" style={{ color: damageColor }}>
                            {result.damage_percent.toFixed(1)}%
                        </div>
                        <div className="stat-label">Lung Damage</div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon confidence">🎯</div>
                    <div>
                        <div className="stat-value" style={{ color: 'var(--success)' }}>
                            {(result.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="stat-label">Confidence</div>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-icon time">⏱️</div>
                    <div>
                        <div className="stat-value" style={{ color: 'var(--warning)' }}>
                            {result.processing_time?.toFixed(1) || '—'}s
                        </div>
                        <div className="stat-label">Processing Time</div>
                    </div>
                </div>
            </div>

            {/* Two-column results */}
            <div className="results-grid">
                {/* CT Viewer with heatmap */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">🖼️ CT Scan Visualization</h3>
                    </div>
                    <div className="ct-viewer">
                        {slices.length > 0 ? (
                            <>
                                <img
                                    src={getHeatmapUrl(scanId, activeSlice)}
                                    alt={`CT Slice ${activeSlice}`}
                                    style={{ width: '100%', imageRendering: 'auto' }}
                                />
                                <div className="ct-viewer-controls">
                                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                        Slice {activeSlice + 1}/{slices.length}
                                    </span>
                                    <input
                                        type="range"
                                        className="slice-slider"
                                        min={0}
                                        max={slices.length - 1}
                                        value={activeSlice}
                                        onChange={(e) => setActiveSlice(parseInt(e.target.value))}
                                    />
                                </div>
                            </>
                        ) : (
                            <div style={{
                                padding: '4rem 2rem',
                                textAlign: 'center',
                                color: 'var(--text-muted)',
                            }}>
                                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🖼️</div>
                                <p>Heatmap slices not available</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Damage gauge & details */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">📊 Damage Assessment</h3>
                    </div>

                    {/* Circular gauge */}
                    <DamageGauge percent={result.damage_percent} color={damageColor} />

                    {/* Severity breakdown */}
                    <div style={{ marginTop: '1.5rem' }}>
                        <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Class Probabilities
                        </h4>
                        {result.probabilities?.classes ? (
                            ['Normal', 'Mild', 'Moderate', 'Severe'].map((label, i) => {
                                const prob = (result.probabilities.classes[i] || 0) * 100
                                return (
                                    <div key={label} style={{ marginBottom: '0.75rem' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                                            <span>{label}</span>
                                            <span style={{ color: SEVERITY_COLORS[label] }}>{prob.toFixed(1)}%</span>
                                        </div>
                                        <div className="progress-bar-bg">
                                            <div
                                                className="progress-bar-fill"
                                                style={{
                                                    width: `${prob}%`,
                                                    background: SEVERITY_COLORS[label],
                                                }}
                                            />
                                        </div>
                                    </div>
                                )
                            })
                        ) : (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                                Probability breakdown not available
                            </p>
                        )}
                    </div>

                    {/* Longitudinal change */}
                    {result.change && (
                        <div style={{
                            marginTop: '1.5rem',
                            padding: '1rem',
                            background: 'rgba(99, 102, 241, 0.06)',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid rgba(99, 102, 241, 0.15)',
                        }}>
                            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>
                                📈 Longitudinal Change
                            </h4>
                            <div style={{ fontSize: '1.3rem', fontWeight: 700 }}>
                                {result.change.change_label === 'Improved' ? '📉' :
                                    result.change.change_label === 'Worsened' ? '📈' : '➡️'}{' '}
                                {result.change.change_label}
                            </div>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                                Score: {result.change.change_score?.toFixed(2) || '—'}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Stage timing breakdown */}
            {result.stage_times && (
                <div className="card" style={{ marginTop: '1.5rem' }}>
                    <div className="card-header">
                        <h3 className="card-title">⏱️ Processing Breakdown</h3>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
                        {Object.entries(result.stage_times).map(([stage, time]) => (
                            <div key={stage} style={{
                                padding: '0.75rem',
                                background: 'var(--bg-input)',
                                borderRadius: 'var(--radius-sm)',
                                textAlign: 'center',
                            }}>
                                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-primary)' }}>
                                    {time.toFixed(2)}s
                                </div>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'capitalize', marginTop: '0.25rem' }}>
                                    {stage}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

function DamageGauge({ percent, color }) {
    const radius = 65
    const circumference = 2 * Math.PI * radius
    const offset = circumference - (percent / 100) * circumference

    return (
        <div className="damage-gauge">
            <svg width="160" height="160" viewBox="0 0 160 160">
                <circle
                    cx="80" cy="80" r={radius}
                    fill="none"
                    stroke="var(--border)"
                    strokeWidth="10"
                />
                <circle
                    cx="80" cy="80" r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="10"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    style={{ transition: 'stroke-dashoffset 1s ease' }}
                />
            </svg>
            <div className="damage-gauge-label">
                <div className="damage-gauge-value" style={{ color }}>
                    {percent.toFixed(1)}
                </div>
                <div className="damage-gauge-unit">% damage</div>
            </div>
        </div>
    )
}
