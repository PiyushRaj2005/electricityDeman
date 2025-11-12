import React from 'react';
import { Calendar, TrendingUp, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Forecast } from '../types';
import { format } from 'date-fns';

interface ForecastHistoryProps {
  forecasts: Forecast[];
  onSelectForecast: (forecast: Forecast) => void;
}

export const ForecastHistory: React.FC<ForecastHistoryProps> = ({
  forecasts,
  onSelectForecast,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = 'px-3 py-1 rounded-full text-xs font-semibold';
    switch (status) {
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-700`;
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-700`;
      case 'pending':
        return `${baseClasses} bg-yellow-100 text-yellow-700`;
      default:
        return baseClasses;
    }
  };

  if (forecasts.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
        <TrendingUp className="w-16 h-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-700 mb-2">No Forecasts Yet</h3>
        <p className="text-gray-500">Create your first forecast to get started</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-6">Forecast History</h3>

      <div className="space-y-4">
        {forecasts.map((forecast) => (
          <div
            key={forecast._id}
            className="border border-gray-200 rounded-xl p-4 hover:shadow-md transition cursor-pointer"
            onClick={() => onSelectForecast(forecast)}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-3">
                {getStatusIcon(forecast.status)}
                <div>
                  <p className="font-semibold text-gray-900">
                    {format(new Date(forecast.timestamp), 'MMM dd, yyyy')}
                  </p>
                  <p className="text-sm text-gray-500">
                    {format(new Date(forecast.timestamp), 'hh:mm a')}
                  </p>
                </div>
              </div>
              <span className={getStatusBadge(forecast.status)}>
                {forecast.status.charAt(0).toUpperCase() + forecast.status.slice(1)}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Temperature</p>
                <p className="font-medium text-gray-900">
                  {forecast.inputData.weatherConditions.temperature}°C
                </p>
              </div>
              <div>
                <p className="text-gray-500">Humidity</p>
                <p className="font-medium text-gray-900">
                  {forecast.inputData.weatherConditions.humidity}%
                </p>
              </div>
              <div>
                <p className="text-gray-500">Wind Speed</p>
                <p className="font-medium text-gray-900">
                  {forecast.inputData.weatherConditions.windSpeed} km/h
                </p>
              </div>
            </div>

            {forecast.modelMetrics && (
              <div className="mt-3 pt-3 border-t border-gray-100 flex space-x-6 text-xs">
                <div>
                  <span className="text-gray-500">R² Score: </span>
                  <span className="font-semibold text-gray-900">
                    {forecast.modelMetrics.r2Score?.toFixed(3)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">RMSE: </span>
                  <span className="font-semibold text-gray-900">
                    {forecast.modelMetrics.rmse?.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">MAE: </span>
                  <span className="font-semibold text-gray-900">
                    {forecast.modelMetrics.mae?.toFixed(2)}
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
