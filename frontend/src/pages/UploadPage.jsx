import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { uploadScan } from '../api'

const PROCESSING_STEPS = [
    { id: 'upload', label: 'Uploading CT scan...' },
    { id: 'preprocess', label: 'Preprocessing & normalization' },
    { id: 'segment', label: 'Lung segmentation (3D U-Net)' },
    { id: 'register', label: 'Image registration (VoxelMorph)' },
    { id: 'classify', label: 'Severity classification (3D ResNet)' },
    { id: 'save', label: 'Storing results' },
]

export default function UploadPage() {
    const navigate = useNavigate()
    const [file, setFile] = useState(null)
    const [patientId, setPatientId] = useState('')
    const [patientName, setPatientName] = useState('')
    const [baselineScanId, setBaselineScanId] = useState('')
    const [processing, setProcessing] = useState(false)
    const [currentStep, setCurrentStep] = useState(-1)
    const [error, setError] = useState(null)

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            setFile(acceptedFiles[0])
            setError(null)
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        multiple: false,
        accept: {
            'application/dicom': ['.dcm'],
            'application/gzip': ['.nii.gz'],
            'application/octet-stream': ['.nii'],
        },
    })

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!file || !patientId.trim()) return

        setProcessing(true)
        setError(null)

        // Simulate step progression
        const stepInterval = setInterval(() => {
            setCurrentStep((prev) => {
                if (prev < PROCESSING_STEPS.length - 1) return prev + 1
                return prev
            })
        }, 2500)
        setCurrentStep(0)

        try {
            const result = await uploadScan(
                file,
                patientId.trim(),
                patientName.trim() || null,
                baselineScanId.trim() || null
            )
            clearInterval(stepInterval)
            setCurrentStep(PROCESSING_STEPS.length - 1)

            // Brief pause to show completion, then navigate
            setTimeout(() => {
                navigate(`/results/${result.scan_id}`)
            }, 800)
        } catch (err) {
            clearInterval(stepInterval)
            setProcessing(false)
            setCurrentStep(-1)
            setError(err.response?.data?.detail || err.message || 'Analysis failed')
        }
    }

    return (
        <div className="fade-in">
            <div className="page-header">
                <h1 className="page-title">Upload CT Scan</h1>
                <p className="page-subtitle">
                    Upload a lung CT scan for AI-powered Post-COVID analysis
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                {/* Left: Upload form */}
                <div className="card">
                    <form onSubmit={handleSubmit}>
                        {/* Drop zone */}
                        <div
                            {...getRootProps()}
                            className={`upload-zone ${isDragActive ? 'active' : ''} ${file ? 'active' : ''}`}
                        >
                            <input {...getInputProps()} />
                            <div className="upload-icon">{file ? '✅' : '📁'}</div>
                            <div className="upload-title">
                                {file ? file.name : 'Drop CT scan here'}
                            </div>
                            <div className="upload-subtitle">
                                {file
                                    ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                                    : 'or click to browse files'}
                            </div>
                            <div className="upload-formats">
                                <span className="format-badge">DICOM (.dcm)</span>
                                <span className="format-badge">NIfTI (.nii)</span>
                                <span className="format-badge">NIfTI (.nii.gz)</span>
                            </div>
                        </div>

                        {/* Patient info */}
                        <div style={{ marginTop: '1.5rem' }}>
                            <div className="form-group">
                                <label className="form-label">Patient ID *</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="e.g. PAT-001"
                                    value={patientId}
                                    onChange={(e) => setPatientId(e.target.value)}
                                    disabled={processing}
                                    required
                                />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Patient Name</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="Optional"
                                    value={patientName}
                                    onChange={(e) => setPatientName(e.target.value)}
                                    disabled={processing}
                                />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Baseline Scan ID (for comparison)</label>
                                <input
                                    type="text"
                                    className="form-input"
                                    placeholder="Optional — for longitudinal analysis"
                                    value={baselineScanId}
                                    onChange={(e) => setBaselineScanId(e.target.value)}
                                    disabled={processing}
                                />
                            </div>
                        </div>

                        {error && (
                            <div style={{
                                padding: '0.75rem 1rem',
                                background: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                borderRadius: 'var(--radius-sm)',
                                color: 'var(--error)',
                                fontSize: '0.9rem',
                                marginBottom: '1rem',
                            }}>
                                ⚠️ {error}
                            </div>
                        )}

                        <button
                            type="submit"
                            className="btn btn-primary"
                            style={{ width: '100%' }}
                            disabled={!file || !patientId.trim() || processing}
                        >
                            {processing ? '🔬 Analyzing...' : '🚀 Upload & Analyze'}
                        </button>
                    </form>
                </div>

                {/* Right: Processing status or info */}
                <div className="card">
                    {processing ? (
                        <>
                            <div className="card-header">
                                <h3 className="card-title">🔬 Processing Pipeline</h3>
                            </div>
                            <div className="processing-steps">
                                {PROCESSING_STEPS.map((step, i) => {
                                    let status = 'pending'
                                    if (i < currentStep) status = 'done'
                                    else if (i === currentStep) status = 'active'

                                    return (
                                        <div key={step.id} className={`processing-step ${status}`}>
                                            <div className={`step-icon ${status}`}>
                                                {status === 'done' ? '✓' : status === 'active' ? '⋯' : '·'}
                                            </div>
                                            <span>{step.label}</span>
                                        </div>
                                    )
                                })}
                            </div>
                        </>
                    ) : (
                        <>
                            <div className="card-header">
                                <h3 className="card-title">📋 Analysis Pipeline</h3>
                            </div>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.8 }}>
                                <p style={{ marginBottom: '1rem' }}>
                                    Our AI system performs comprehensive Post-COVID lung analysis:
                                </p>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <InfoItem icon="1️⃣" text="DICOM/NIfTI preprocessing & normalization" />
                                    <InfoItem icon="2️⃣" text="3D U-Net lung segmentation" />
                                    <InfoItem icon="3️⃣" text="VoxelMorph deformable image registration" />
                                    <InfoItem icon="4️⃣" text="3D ResNet severity classification" />
                                    <InfoItem icon="5️⃣" text="Longitudinal change detection" />
                                </div>

                                <div style={{
                                    marginTop: '1.5rem',
                                    padding: '1rem',
                                    background: 'rgba(99, 102, 241, 0.06)',
                                    borderRadius: 'var(--radius-sm)',
                                    border: '1px solid rgba(99, 102, 241, 0.15)',
                                }}>
                                    <strong style={{ color: 'var(--accent-primary)' }}>🎯 Output:</strong>
                                    <ul style={{ marginTop: '0.5rem', paddingLeft: '1.25rem' }}>
                                        <li>Severity: Normal / Mild / Moderate / Severe</li>
                                        <li>Lung damage percentage (0–100%)</li>
                                        <li>Segmentation overlay & heatmaps</li>
                                        <li>Longitudinal change tracking</li>
                                    </ul>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}

function InfoItem({ icon, text }) {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span>{icon}</span>
            <span>{text}</span>
        </div>
    )
}
