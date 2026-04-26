import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { RefreshCw, Activity, Terminal, Stethoscope } from 'lucide-react';
import PatientCard from './components/PatientCard';
import ChatInterface from './components/ChatInterface';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [patientData, setPatientData] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState('idle');

  // Poll model status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const r = await axios.get(`${API_BASE}/model_status`);
        setModelStatus(r.data.status);
      } catch { /* ignore */ }
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const resetSession = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/reset`);
      const { patient, report } = response.data;
      setPatientData(patient);
      setReportData(report);
      setMessages([{
        role: 'system',
        text: `New patient case loaded. You are ${patient.name}, a ${patient.age}-year-old who speaks ${patient.language}. Your diagnosis: ${report.diagnosis_name}. Ask the doctor any questions about your condition.`
      }]);
    } catch (error) {
      console.error('Error resetting:', error);
      alert('Failed to connect to backend. Is the server running on port 8000?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (text) => {
    setIsLoading(true);
    // Add patient message to UI
    setMessages(prev => [...prev, { role: 'patient', text }]);

    try {
      const response = await axios.post(`${API_BASE}/chat`, { message: text }, { timeout: 300000 });
      if (response.data.response) {
        setMessages(prev => [...prev, { role: 'doctor', text: response.data.response }]);
      } else if (response.data.error) {
        setMessages(prev => [...prev, { role: 'system', text: `⚠️ ${response.data.error}` }]);
      }
    } catch (error) {
      console.error('Error in chat:', error);
      if (error.code === 'ECONNABORTED') {
        setMessages(prev => [...prev, { role: 'system', text: '⏳ Model is still loading (downloading ~6GB). Please wait and try again.' }]);
      } else {
        setMessages(prev => [...prev, { role: 'system', text: '❌ Failed to get response from the doctor.' }]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    resetSession();
  }, []);

  const statusColor = modelStatus === 'ready' ? '#22c55e' : modelStatus === 'loading' ? '#f59e0b' : '#94a3b8';
  const statusText = modelStatus === 'ready' ? 'Model Ready' : modelStatus === 'loading' ? 'Model Loading...' : 'Model Idle';

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">
          <Stethoscope size={28} style={{ display: 'inline', marginRight: '0.5rem' }} />
          MedBridge AI
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn btn-secondary" onClick={resetSession} disabled={isLoading}>
            <RefreshCw size={18} className={isLoading ? 'animate-spin' : ''} />
            New Patient
          </button>
          <div className="glass-card" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: statusColor, boxShadow: `0 0 8px ${statusColor}` }} />
            {statusText}
          </div>
        </div>
      </header>

      <main style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <ChatInterface 
          messages={messages} 
          onSendMessage={handleSendMessage} 
          isLoading={isLoading}
        />
      </main>

      <aside style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        {/* Patient Info Card */}
        {patientData && reportData && (
          <div className="glass-card animate-fade-in">
            <div className="patient-header">
              <div className="avatar">
                {patientData.name ? patientData.name[0] : '?'}
              </div>
              <div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>You are: {patientData.name}</h2>
                <p className="text-muted" style={{ fontSize: '0.875rem' }}>
                  {patientData.age}y • {patientData.gender} • {patientData.language}
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={18} className="text-muted" />
                  <span style={{ fontWeight: 600 }}>Your Diagnosis</span>
                </div>
                <span className={`badge ${reportData.severity === 'Critical' ? 'badge-high' : reportData.severity === 'Moderate' ? 'badge-medium' : 'badge-low'}`}>
                  {reportData.severity}
                </span>
              </div>
              <p style={{ fontSize: '1rem', fontWeight: 500, color: '#e2e8f0' }}>{reportData.diagnosis_name}</p>

              <div style={{ marginTop: '0.5rem' }}>
                <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>Medical Report</span>
                <div className="glass-card" style={{ 
                  background: 'rgba(0,0,0,0.2)', padding: '1rem', fontSize: '0.875rem', 
                  lineHeight: 1.6, maxHeight: '200px', overflowY: 'auto', marginTop: '0.5rem'
                }}>
                  {reportData.medical_report}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* How to use */}
        <div className="glass-card">
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>💡 How to Use</h3>
          <ul style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.8, paddingLeft: '1rem' }}>
            <li>You are the <strong>patient</strong></li>
            <li>The AI model is your <strong>doctor</strong></li>
            <li>Ask about your diagnosis, treatment, diet, etc.</li>
            <li>The doctor responds in your language</li>
            <li>Click "New Patient" for a different case</li>
          </ul>
        </div>
      </aside>

      <footer style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '2rem', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
        MedBridge AI — Compassionate Medical Communication • Powered by GRPO-trained Qwen 2.5
      </footer>
    </div>
  );
}

export default App;
