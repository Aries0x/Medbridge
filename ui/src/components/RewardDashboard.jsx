import React from 'react';
import { Award, CheckCircle2, XCircle, Info } from 'lucide-react';

const RewardDashboard = ({ observation }) => {
  if (!observation || !observation.reward_breakdown) return null;

  const { reward_breakdown, reward } = observation;

  return (
    <div className="glass-card animate-fade-in" style={{ height: 'fit-content' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
        <Award size={24} className="text-primary" />
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Evaluation Results</h2>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <div style={{ fontSize: '3rem', fontWeight: 800, color: reward > 0 ? 'var(--success)' : 'var(--error)' }}>
          {reward?.toFixed(2)}
        </div>
        <p className="text-muted">Total Score</p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {Object.entries(reward_breakdown).map(([key, value]) => (
          <div key={key} className="glass-card" style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
              <span style={{ fontSize: '0.875rem', fontWeight: 600, textTransform: 'capitalize' }}>
                {key.replace(/_/g, ' ')}
              </span>
              <span style={{ fontWeight: 700, color: value > 0 ? 'var(--success)' : 'var(--error)' }}>
                {value > 0 ? `+${value}` : value}
              </span>
            </div>
            <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ 
                width: `${Math.min(Math.abs(value) * 100, 100)}%`, 
                height: '100%', 
                background: value > 0 ? 'var(--success)' : 'var(--error)',
                transition: 'width 0.5s ease-out'
              }} />
            </div>
          </div>
        ))}
      </div>

      {observation.metadata && observation.metadata.evaluation && (
        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '0.75rem', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--primary)' }}>
            <Info size={16} />
            <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>Expert Feedback</span>
          </div>
          <p style={{ fontSize: '0.875rem', lineHeight: 1.5, fontStyle: 'italic' }}>
            {observation.metadata.evaluation}
          </p>
        </div>
      )}
    </div>
  );
};

export default RewardDashboard;
