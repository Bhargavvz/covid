import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getPatients } from '../api'

export default function PatientsPage() {
    const [patients, setPatients] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        loadPatients()
    }, [])

    const loadPatients = async () => {
        try {
            setLoading(true)
            const data = await getPatients()
            setPatients(data)
        } catch (err) {
            setError(err.message || 'Failed to load patients')
        } finally {
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Loading patients...</p>
            </div>
        )
    }

    return (
        <div className="fade-in">
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                    <h1 className="page-title">Patients</h1>
                    <p className="page-subtitle">{patients.length} patient{patients.length !== 1 ? 's' : ''} registered</p>
                </div>
                <Link to="/" className="btn btn-primary">📤 Upload New Scan</Link>
            </div>

            {error && (
                <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--error)',
                    marginBottom: '1rem',
                }}>
                    ⚠️ {error}
                </div>
            )}

            {patients.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">👥</div>
                    <div className="empty-state-title">No Patients Yet</div>
                    <p>Upload a CT scan to create the first patient record.</p>
                    <Link to="/" className="btn btn-primary" style={{ marginTop: '1rem' }}>
                        Upload CT Scan
                    </Link>
                </div>
            ) : (
                <div className="card">
                    <table className="patient-table">
                        <thead>
                            <tr>
                                <th>Patient ID</th>
                                <th>Name</th>
                                <th>Scans</th>
                                <th>Registered</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {patients.map((patient) => (
                                <tr key={patient.id}>
                                    <td>
                                        <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>
                                            {patient.patient_id}
                                        </span>
                                    </td>
                                    <td>{patient.patient_name || '—'}</td>
                                    <td>
                                        <span style={{
                                            background: 'rgba(99, 102, 241, 0.1)',
                                            padding: '0.2rem 0.6rem',
                                            borderRadius: '999px',
                                            fontSize: '0.8rem',
                                            fontWeight: 600,
                                            color: 'var(--accent-primary)',
                                        }}>
                                            {patient.scan_count}
                                        </span>
                                    </td>
                                    <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                        {new Date(patient.created_at).toLocaleDateString()}
                                    </td>
                                    <td>
                                        <Link
                                            to={`/patients/${patient.patient_id}`}
                                            className="btn btn-secondary"
                                            style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                                        >
                                            View History →
                                        </Link>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
