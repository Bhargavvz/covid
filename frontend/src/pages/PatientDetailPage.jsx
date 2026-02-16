import React, { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Line } from 'react-chartjs-2'
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from 'chart.js'
import { getPatientScans, getDamageHistory } from '../api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const SEVERITY_COLORS = {
    Normal: '#10b981',
    Mild: '#f59e0b',
    Moderate: '#f97316',
    Severe: '#ef4444',
}

export default function PatientDetailPage() {
    const { patientId } = useParams()
    const [patient, setPatient] = useState(null)
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        loadPatientData()
    }, [patientId])

    const loadPatientData = async () => {
        try {
            setLoading(true)
            const [patientData, historyData] = await Promise.all([
                getPatientScans(patientId).catch(() => null),
                getDamageHistory(patientId).catch(() => ({ history: [] })),
            ])
            setPatient(patientData)
            setHistory(historyData.history || [])
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Loading patient data...</p>
            </div>
        )
    }

    if (!patient) {
        return (
            <div className="empty-state">
                <div className="empty-state-icon">🔍</div>
                <div className="empty-state-title">Patient Not Found</div>
                <Link to="/patients" className="btn btn-primary" style={{ marginTop: '1rem' }}>
                    Back to Patients
                </Link>
            </div>
        )
    }

    // Chart data
    const chartData = {
        labels: history.map((h, i) =>
            h.study_date || `Scan ${i + 1}`
        ),
        datasets: [
            {
                label: 'Lung Damage %',
                data: history.map((h) => h.damage_percent),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: history.map(
                    (h) => SEVERITY_COLORS[h.severity_label] || '#6366f1'
                ),
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8,
            },
        ],
    }

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                titleFont: { family: 'Inter' },
                bodyFont: { family: 'Inter' },
                padding: 12,
                cornerRadius: 8,
                callbacks: {
                    label: (ctx) => {
                        const h = history[ctx.dataIndex]
                        return [
                            `Damage: ${h.damage_percent.toFixed(1)}%`,
                            `Severity: ${h.severity_label}`,
                        ]
                    },
                },
            },
        },
        scales: {
            y: {
                min: 0,
                max: 100,
                title: { display: true, text: 'Lung Damage %', color: '#94a3b8' },
                grid: { color: 'rgba(42, 48, 80, 0.5)' },
                ticks: { color: '#94a3b8' },
            },
            x: {
                grid: { color: 'rgba(42, 48, 80, 0.3)' },
                ticks: { color: '#94a3b8' },
            },
        },
    }

    return (
        <div className="fade-in">
            <div className="page-header">
                <Link to="/patients" style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    ← Back to Patients
                </Link>
                <h1 className="page-title" style={{ marginTop: '0.5rem' }}>
                    {patient.patient_name || patient.patient_id}
                </h1>
                <p className="page-subtitle">
                    Patient ID: {patient.patient_id} · {patient.scan_count} scan{patient.scan_count !== 1 ? 's' : ''}
                </p>
            </div>

            {/* Damage over time chart */}
            {history.length > 0 && (
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                    <div className="card-header">
                        <h3 className="card-title">📈 Lung Damage Over Time</h3>
                    </div>
                    <div style={{ height: '300px' }}>
                        <Line data={chartData} options={{ ...chartOptions, maintainAspectRatio: false }} />
                    </div>
                </div>
            )}

            {/* Scan history */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">🗂️ Scan History</h3>
                </div>

                {patient.scans && patient.scans.length > 0 ? (
                    <table className="patient-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Type</th>
                                <th>Severity</th>
                                <th>Damage</th>
                                <th>Change</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {patient.scans.map((scan) => (
                                <tr key={scan.id}>
                                    <td style={{ fontWeight: 500 }}>
                                        {scan.study_date || new Date(scan.uploaded_at).toLocaleDateString()}
                                    </td>
                                    <td>
                                        <span className="format-badge">{scan.file_type.toUpperCase()}</span>
                                    </td>
                                    <td>
                                        {scan.result ? (
                                            <span className={`severity-badge ${scan.result.severity_label.toLowerCase()}`}>
                                                {scan.result.severity_label}
                                            </span>
                                        ) : '—'}
                                    </td>
                                    <td style={{ fontWeight: 600 }}>
                                        {scan.result ? `${scan.result.damage_percent.toFixed(1)}%` : '—'}
                                    </td>
                                    <td>
                                        {scan.result?.change ? (
                                            <span style={{
                                                color: scan.result.change.change_label === 'Improved'
                                                    ? 'var(--success)'
                                                    : scan.result.change.change_label === 'Worsened'
                                                        ? 'var(--error)'
                                                        : 'var(--text-muted)',
                                                fontWeight: 500,
                                            }}>
                                                {scan.result.change.change_label}
                                            </span>
                                        ) : '—'}
                                    </td>
                                    <td>
                                        <Link
                                            to={`/results/${scan.id}`}
                                            className="btn btn-secondary"
                                            style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                                        >
                                            Details →
                                        </Link>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <div className="empty-state" style={{ padding: '2rem' }}>
                        <p>No scans recorded for this patient.</p>
                    </div>
                )}
            </div>
        </div>
    )
}
