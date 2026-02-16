import React from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'
import PatientsPage from './pages/PatientsPage'
import PatientDetailPage from './pages/PatientDetailPage'

function App() {
    return (
        <BrowserRouter>
            <div className="app-layout">
                <Navbar />
                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<UploadPage />} />
                        <Route path="/results/:scanId" element={<ResultsPage />} />
                        <Route path="/patients" element={<PatientsPage />} />
                        <Route path="/patients/:patientId" element={<PatientDetailPage />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    )
}

function Navbar() {
    return (
        <nav className="navbar">
            <div className="navbar-brand">
                <div className="logo-icon">🫁</div>
                <span>CT Analysis</span>
            </div>

            <div className="navbar-links">
                <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    📤 Upload
                </NavLink>
                <NavLink to="/patients" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    👥 Patients
                </NavLink>
            </div>

            <div className="gpu-badge">
                <span className="dot"></span>
                GPU Ready
            </div>
        </nav>
    )
}

export default App
