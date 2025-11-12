import React, { useState } from 'react';
import { Cloud, Calendar, Send } from 'lucide-react';
import { forecastAPI } from '../services/api';
import { Forecast } from '../types';

interface ForecastFormProps {
  onForecastCreated: (forecast: Forecast) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export const ForecastForm: React.FC<ForecastFormProps> = ({
  onForecastCreated,
  isLoading,
  setIsLoading,
}) => {
  const [temperature, setTemperature] = useState(25);
  const [humidity, setHumidity] = useState(60);
  const [windSpeed, setWindSpeed] = useState(10);
  const [isHoliday, setIsHoliday] = useState(false);
  const [isWeekend, setIsWeekend] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await forecastAPI.createForecast({
        startDate: new Date().toISOString(),
        endDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
        weatherConditions: {
          temperature,
          humidity,
          windSpeed,
        },
        isHoliday,
        isWeekend,
        forecastHorizon: 'all',
      });

      onForecastCreated(response.data.forecast);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to create forecast');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-6">Input Parameters</h3>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Cloud className="w-4 h-4 inline mr-1" />
              Temperature (°C)
            </label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              min="-10"
              max="50"
              step="0.1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Cloud className="w-4 h-4 inline mr-1" />
              Humidity (%)
            </label>
            <input
              type="number"
              value={humidity}
              onChange={(e) => setHumidity(Number(e.target.value))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              min="0"
              max="100"
              step="1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Cloud className="w-4 h-4 inline mr-1" />
              Wind Speed (km/h)
            </label>
            <input
              type="number"
              value={windSpeed}
              onChange={(e) => setWindSpeed(Number(e.target.value))}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              min="0"
              max="100"
              step="0.1"
            />
          </div>
        </div>

        <div className="flex space-x-6">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={isHoliday}
              onChange={(e) => setIsHoliday(e.target.checked)}
              className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700">
              <Calendar className="w-4 h-4 inline mr-1" />
              Holiday
            </span>
          </label>

          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={isWeekend}
              onChange={(e) => setIsWeekend(e.target.checked)}
              className="w-5 h-5 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700">
              <Calendar className="w-4 h-4 inline mr-1" />
              Weekend
            </span>
          </label>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          <Send className="w-5 h-5" />
          <span>{isLoading ? 'Generating Forecast...' : 'Generate Forecast'}</span>
        </button>
      </form>
    </div>
  );
};
