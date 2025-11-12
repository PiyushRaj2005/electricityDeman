import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { LogOut, Zap, TrendingUp, Calendar, Cloud } from 'lucide-react';
import { ForecastForm } from './ForecastForm';
import { ForecastResults } from './ForecastResults';
import { ForecastHistory } from './ForecastHistory';
import { Forecast } from '../types';
import { forecastAPI } from '../services/api';

export const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<'forecast' | 'history'>('forecast');
  const [currentForecast, setCurrentForecast] = useState<Forecast | null>(null);
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (activeTab === 'history') {
      loadForecasts();
    }
  }, [activeTab]);

  const loadForecasts = async () => {
    try {
      const response = await forecastAPI.getUserForecasts();
      setForecasts(response.data.forecasts);
    } catch (error) {
      console.error('Failed to load forecasts:', error);
    }
  };

  const handleForecastCreated = (forecast: Forecast) => {
    setCurrentForecast(forecast);
    setForecasts((prev) => [forecast, ...prev]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">CSSM Forecasting</h1>
                <p className="text-xs text-gray-500">Cross-Scale Sequence Mixer</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-700">{user?.name}</p>
                <p className="text-xs text-gray-500">{user?.email}</p>
              </div>
              <button
                onClick={logout}
                className="flex items-center space-x-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm font-medium">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Electricity Demand Forecasting</h2>
              <p className="text-gray-600 mt-1">Multi-horizon prediction using deep learning</p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('forecast')}
                className={`px-6 py-2 rounded-lg font-medium transition ${
                  activeTab === 'forecast'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <TrendingUp className="w-4 h-4 inline mr-2" />
                New Forecast
              </button>
              <button
                onClick={() => setActiveTab('history')}
                className={`px-6 py-2 rounded-lg font-medium transition ${
                  activeTab === 'history'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Calendar className="w-4 h-4 inline mr-2" />
                History
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-600 font-medium">Model Accuracy</p>
                  <p className="text-2xl font-bold text-blue-900 mt-1">92.5%</p>
                </div>
                <TrendingUp className="w-8 h-8 text-blue-600" />
              </div>
            </div>

            <div className="bg-green-50 rounded-xl p-4 border border-green-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-600 font-medium">Total Forecasts</p>
                  <p className="text-2xl font-bold text-green-900 mt-1">{forecasts.length}</p>
                </div>
                <Calendar className="w-8 h-8 text-green-600" />
              </div>
            </div>

            <div className="bg-cyan-50 rounded-xl p-4 border border-cyan-100">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-cyan-600 font-medium">Time Scales</p>
                  <p className="text-2xl font-bold text-cyan-900 mt-1">4</p>
                </div>
                <Cloud className="w-8 h-8 text-cyan-600" />
              </div>
            </div>
          </div>
        </div>

        {activeTab === 'forecast' ? (
          <div className="space-y-6">
            <ForecastForm
              onForecastCreated={handleForecastCreated}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
            {currentForecast && <ForecastResults forecast={currentForecast} />}
          </div>
        ) : (
          <ForecastHistory forecasts={forecasts} onSelectForecast={setCurrentForecast} />
        )}
      </div>
    </div>
  );
};
