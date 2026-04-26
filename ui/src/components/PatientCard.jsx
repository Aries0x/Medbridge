import React from 'react';
import { User, FileText, Activity } from 'lucide-react';

const PatientCard = ({ observation }) => {
  if (!observation || !observation.patient_profile) return null;

  const { patient_profile, medical_report, diagnosis_name, severity } = observation;
  const severityClass = severity?.toLowerCase() === 'high' ? 'badge-high' : 
                        severity?.toLowerCase() === 'medium' ? 'badge-medium' : 'badge-low';

  return (
    <div className="glass-card animate-fade-in">
      <div className="patient-header">
        <div className="avatar">
          {patient_profile.name ? patient_profile.name[0] : '?'}
        </div>
        <div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>{patient_profile.name || 'Unknown Patient'}</h2>
          <p className="text-muted" style={{ fontSize: '0.875rem' }}>
            {patient_profile.age}y • {patient_profile.gender} • {patient_profile.occupation}
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} className="text-muted" />
            <span style={{ fontWeight: 600 }}>Diagnosis</span>
          </div>
          <span className={`badge ${severityClass}`}>{severity}</span>
        </div>
        <p style={{ fontSize: '1rem', fontWeight: 500, color: '#e2e8f0' }}>{diagnosis_name}</p>

        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <FileText size={18} className="text-muted" />
            <span style={{ fontWeight: 600 }}>Medical Report</span>
          </div>
          <div className="glass-card" style={{ 
            background: 'rgba(0,0,0,0.2)', 
            padding: '1rem', 
            fontSize: '0.875rem', 
            lineHeight: 1.6,
            maxHeight: '300px',
            overflowY: 'auto'
          }}>
            {medical_report}
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '1rem', marginTop: '0.5rem' }}>
          <h3 style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>History & Lifestyle</h3>
          <ul style={{ listSetyle: 'none', fontSize: '0.875rem', color: '#cbd5e1' }}>
            <li><span style={{ color: 'var(--text-muted)' }}>History:</span> {patient_profile.medical_history}</li>
            <li style={{ marginTop: '0.5rem' }}><span style={{ color: 'var(--text-muted)' }}>Lifestyle:</span> {patient_profile.lifestyle}</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default PatientCard;
