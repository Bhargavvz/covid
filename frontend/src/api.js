import axios from 'axios'

const api = axios.create({
    baseURL: '/api',
    timeout: 300000, // 5 min for large scans
})

export const uploadScan = async (file, patientId, patientName, baselineScanId) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('patient_id', patientId)
    if (patientName) formData.append('patient_name', patientName)
    if (baselineScanId) formData.append('baseline_scan_id', baselineScanId)

    const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
            if (e.total) {
                const pct = Math.round((e.loaded * 100) / e.total)
                console.log(`Upload: ${pct}%`)
            }
        },
    })
    return response.data
}

export const getResult = async (scanId) => {
    const response = await api.get(`/results/${scanId}`)
    return response.data
}

export const getResultSummary = async (scanId) => {
    const response = await api.get(`/results/${scanId}/summary`)
    return response.data
}

export const getHeatmapUrl = (scanId, sliceIdx = 0) =>
    `/api/results/${scanId}/heatmap?slice_idx=${sliceIdx}`

export const getSlices = async (scanId) => {
    const response = await api.get(`/results/${scanId}/slices`)
    return response.data
}

export const getPatients = async (skip = 0, limit = 50) => {
    const response = await api.get('/patients', { params: { skip, limit } })
    return response.data
}

export const getPatientScans = async (patientId) => {
    const response = await api.get(`/patients/${patientId}/scans`)
    return response.data
}

export const getDamageHistory = async (patientId) => {
    const response = await api.get(`/patients/${patientId}/damage-history`)
    return response.data
}

export const healthCheck = async () => {
    const response = await api.get('/health')
    return response.data
}

export default api
