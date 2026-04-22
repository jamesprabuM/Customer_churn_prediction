import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [customerData, setCustomerData] = useState({
    gender: 'Female', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'No',
    tenure: 12, PhoneService: 'Yes', MultipleLines: 'No',
    InternetService: 'Fiber optic', OnlineSecurity: 'No',
    OnlineBackup: 'No', DeviceProtection: 'No', TechSupport: 'No',
    StreamingTV: 'Yes', StreamingMovies: 'No', Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check',
    MonthlyCharges: 70.0, TotalCharges: '840.0'
  });

  const [prediction, setPrediction] = useState(null);
  const [segment, setSegment] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [humanExplanation, setHumanExplanation] = useState(null);
  const [cost, setCost] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [driftData, setDriftData] = useState(null);
  const [isVideoOpen, setIsVideoOpen] = useState(false);

  useEffect(() => {
    // Mock API call for model drift monitoring data
    setTimeout(() => {
      setDriftData({
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        datasets: [
          {
            label: 'Model Accuracy',
            data: [0.89, 0.88, 0.87, 0.88, 0.85, 0.83, 0.81],
            borderColor: 'rgba(59, 130, 246, 1)',
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Accuracy Threshold',
            data: [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
            borderColor: 'rgba(239, 68, 68, 1)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          }
        ]
      });
    }, 1000);
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    
    let parsedValue = value;
    if (['SeniorCitizen', 'tenure'].includes(name)) {
      // Allow empty string to let user delete numbers clearly
      parsedValue = value === '' ? '' : parseInt(value);
    } else if (name === 'MonthlyCharges') {
      parsedValue = value === '' ? '' : parseFloat(value);
    }
    
    setCustomerData(prev => ({ 
        ...prev, 
        [name]: parsedValue 
    }));
  };

  const handlePredict = async () => {
    // Basic validation before sending
    if (customerData.tenure === '' || customerData.MonthlyCharges === '') {
      setError('Please enter valid numbers for Tenure and Monthly Charges.');
      return;
    }
    setLoading(true);
    setError(null);
    setPrediction(null);
    setSegment(null);
    setExplanation(null);
    setHumanExplanation(null);
    setCost(null);
    
    try {
      const baseUrl = 'http://127.0.0.1:8000';
      
      const predRes = await axios.post(baseUrl + '/predict', customerData);
      setPrediction(predRes.data.churn_probability);
      
      const segRes = await axios.post(baseUrl + '/segment', customerData);
      setSegment(segRes.data.recommended_action);

      const expRes = await axios.post(baseUrl + '/explain', customerData);
      setExplanation(expRes.data.top_factors);
      setHumanExplanation(expRes.data.human_explanation);

      const costRes = await axios.post(baseUrl + '/cost', customerData);
      setCost(costRes.data);
      
    } catch (err) {
      console.error(err);
      setError('Failed to fetch prediction. Ensure FastAPI backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (prob) => {
    if (prob > 0.7) return '#ef4444';
    if (prob > 0.4) return '#f59e0b';
    return '#10b981';
  };

  const shapChartData = explanation ? {
    labels: Object.keys(explanation),
    datasets: [
      {
        label: 'SHAP Value (Impact)',
        data: Object.values(explanation),
        backgroundColor: Object.values(explanation).map(v => 
          v > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.7)'
        ),
        borderRadius: 4,
      },
    ],
  } : null;

  const shapChartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Feature Impact on Churn', font: { size: 16 } },
    },
  };

  // Framer Motion Variants
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };
  
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 100 } }
  };

  return (
    <>
      <div className="animated-bg"></div>
      <div style={{ display: 'flex', height: '100vh', width: '100vw', background: 'transparent', position: 'relative', zIndex: 1, fontFamily: '"Inter", sans-serif', color: '#fcfaf7', overflow: 'hidden' }}>
        
        {/* MAIN LAYOUT */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          
          {/* TOP NAVBAR */}
          <div style={{ padding: '2rem 3rem', zIndex: 5, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: '800', fontFamily: '"Space Grotesk", sans-serif', background: 'linear-gradient(90deg, #fb923c 0%, #ffedd5 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.05em' }}>Customer Churn Prediction</h1>
            </div>
            {/* Business Context */}
            <div style={{ color: '#d6d3d1', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ border: '1px solid #fb923c', color: '#fb923c', padding: '3px 10px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold' }}>TELEMETRY.NODE</span>
              Identify at-risk units using predictive topology.
            </div>
          </div>

          {/* DASHBOARD CONTENT */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
            style={{ padding: '3rem', maxWidth: '1400px', margin: '0 auto', width: '100%', boxSizing: 'border-box' }}
          >
            <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
              
              {/* LEFT INPUT PANEL */}
              <motion.div 
                className="bento-card"
                style={{ flex: '1 1 350px', display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '2rem' }}
              >
                <h2 style={{ color: '#fcfaf7', margin: '0 0 2rem 0', fontSize: '1.3rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontFamily: '"Space Grotesk", sans-serif' }}>[ SIMULATOR_INPUT ]</h2>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
                  <label style={{ fontWeight: '600', color: '#a8a29e', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '2px' }}>Tenure Range</label>
                  <input type='number' name='tenure' className='abstract-input' value={customerData.tenure} onChange={handleInputChange} />
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center', marginTop: '1rem' }}>
                  <label style={{ fontWeight: '600', color: '#a8a29e', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '2px' }}>Financial Outflow (₹)</label>
                  <input type='number' step='0.01' name='MonthlyCharges' className='abstract-input' value={customerData.MonthlyCharges} onChange={handleInputChange} />
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center', marginTop: '1rem' }}>
                  <label style={{ fontWeight: '600', color: '#a8a29e', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '2px' }}>Topology Config</label>
                  <select name='Contract' className='abstract-input' value={customerData.Contract} onChange={handleInputChange} style={{ WebkitAppearance: 'none', background: 'transparent' }}>
                    <option style={{ background: '#404b3f' }}>Month-to-month</option><option style={{ background: '#404b3f' }}>One year</option><option style={{ background: '#404b3f' }}>Two year</option>
                  </select>
                </div>
                
                <motion.button 
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(251, 146, 60, 0.1)' }} whileTap={{ scale: 0.98 }}
                  onClick={handlePredict} disabled={loading} 
                  style={{ marginTop: '3rem', padding: '1rem', background: 'transparent', color: '#fb923c', border: '1px solid #fb923c', borderRadius: '4px', cursor: 'pointer', fontFamily: '"Space Grotesk", sans-serif', fontWeight: 'bold', fontSize: '1rem', letterSpacing: '2px', textTransform: 'uppercase' }}
                >
                  {loading ? 'Evaluating Model...' : 'Execute Analysis'}
                </motion.button>
              </motion.div>
              
              {/* RIGHT RESULTS PANEL */}
              <div style={{ flex: '1 1 600px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <AnimatePresence mode="wait">
                    {error && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ padding: '1.5rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', color: '#ef4444', borderRadius: '12px' }}>
                        ⚠️ {error}
                      </motion.div>
                    )}
                    
                    {!prediction && !loading && !error && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '16px', border: '2px dashed #475569', color: '#64748b', fontSize: '1.2rem' }}>
                           Awaiting telemetry inputs...
                        </motion.div>
                    )}

                    {prediction !== null && !loading && (
                      <motion.div variants={containerVariants} initial="hidden" animate="show" style={{ display: 'flex', gap: '1.5rem', flexDirection: 'column' }}>
                        
                        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                          <motion.div variants={itemVariants} className="bento-card" style={{ flex: '1', padding: '2rem', borderTop: '4px solid ' + (prediction > 0.6 ? '#ef4444' : '#fb923c') }}>
                            <h3 style={{ margin: 0, color: '#a8a29e', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px', textAlign: 'center' }}>AI Confidence Score</h3>
                            <div style={{ fontSize: '4.5rem', fontFamily: '"Space Grotesk", sans-serif', fontWeight: '900', color: (prediction > 0.6 ? '#ef4444' : '#fb923c'), lineHeight: 1.2, marginTop: '1rem', textAlign: 'center' }}>
                              {(prediction * 100).toFixed(1)}%
                            </div>
                          </motion.div>
                          
                          {/* Segment Card */}
                          {segment && (
                            <motion.div variants={itemVariants} className="bento-card glow-border" style={{ flex: '1', padding: '2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center', textAlign: 'center' }}>
                                <h3 style={{ margin: 0, color: '#a8a29e', fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Prescriptive Target</h3>
                                <div style={{ marginTop: '1.5rem', color: '#fb923c', fontWeight: '600', fontSize: '1.4rem', lineHeight: '1.4', fontFamily: '"Space Grotesk", sans-serif' }}>{segment}</div>
                            </motion.div>
                          )}
                        </div>

                        {/* Chart / LLM Explainer */}
                        {explanation && shapChartData && (
                          <motion.div variants={itemVariants} className="bento-card" style={{ padding: '2rem' }}>
                            {humanExplanation && (
                              <div style={{ marginBottom: '2rem', padding: '1.5rem', border: '1px dashed rgba(255, 255, 255, 0.1)', color: '#d6d3d1', fontSize: '1.1rem', fontWeight: '400', lineHeight: 1.6 }}>
                                <strong style={{ color: '#a8a29e', display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '2px' }}>[ LLM INTERPRETER ]</strong>
                                {humanExplanation}
                              </div>
                            )}
                            <div style={{ height: '280px', width: '100%' }}>
                              <Bar 
                                data={shapChartData} 
                                options={{
                                  ...shapChartOptions, 
                                  plugins: { legend: { display: false }, title: { display: true, text: 'SHAP Feature Contribution Matrix', color: '#94a3b8', font: { size: 14 } } },
                                  scales: { x: { ticks: { color: '#64748b' }, grid: { color: '#334155' } }, y: { ticks: { color: '#cbd5e1' }, grid: { display: false } } }
                                }} 
                              />
                            </div>
                          </motion.div>
                        )}

                        {/* Financial Analytics */}
                        {cost && (
                          <motion.div variants={itemVariants} className="bento-card glow-border" style={{ padding: '2rem' }}>
                            <h3 style={{ marginTop: 0, marginBottom: '2rem', color: '#fcfaf7', fontFamily: '"Space Grotesk", sans-serif', fontSize: '1.2rem', textTransform: 'uppercase', letterSpacing: '2px', textAlign: 'center' }}>[ EXECUTIVE FINANCIAL ARRAY ]</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
                              <div className="bento-card glow-alert" style={{ padding: '1.5rem', overflow: 'hidden', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#ef4444', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '2px', whiteSpace: 'nowrap' }}>Risk Exposure</div>
                                <div style={{ fontSize: '2.5rem', fontWeight: '900', color: '#ef4444', fontFamily: '"Space Grotesk", sans-serif', marginTop: '1rem', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>₹{Math.round(cost.potential_loss || 0).toLocaleString()}</div>
                              </div>
                              <div className="bento-card" style={{ padding: '1.5rem', overflow: 'hidden', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#a8a29e', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '2px', whiteSpace: 'nowrap' }}>Offer Node</div>
                                <div style={{ fontSize: '2.5rem', fontWeight: '900', color: '#a8a29e', fontFamily: '"Space Grotesk", sans-serif', marginTop: '1rem', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>₹{Math.round(cost.retention_cost || 0).toLocaleString()}</div>
                              </div>
                              <div className="bento-card glow-border" style={{ padding: '1.5rem', overflow: 'hidden', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.85rem', color: '#fb923c', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '2px', whiteSpace: 'nowrap' }}>Net Impact</div>
                                <div style={{ fontSize: '2.5rem', fontWeight: '900', color: '#fb923c', fontFamily: '"Space Grotesk", sans-serif', marginTop: '1rem', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>₹{Math.round(cost.net_savings || 0).toLocaleString()}</div>
                              </div>
                            </div>
                            <div style={{ padding: '1.25rem 1.5rem', background: 'rgba(251, 146, 60, 0.05)', border: '1px dashed rgba(251, 146, 60, 0.3)', borderRadius: '4px', color: '#e7e5e4', fontWeight: '500', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                             {cost.recommendation}
                            </div>
                          </motion.div>
                        )}

                      </motion.div>
                    )}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </>
  );
}

export default App;
