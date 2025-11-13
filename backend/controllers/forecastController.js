import axios from 'axios';
import Forecast from '../models/Forecast.js';

export const createForecast = async (req, res) => {
  try {
    const { startDate, endDate, weatherConditions, isHoliday, isWeekend, forecastHorizon } = req.body;

    const forecast = await Forecast.create({
      userId: req.user._id,
      inputData: {
        startDate,
        endDate,
        weatherConditions,
        isHoliday: isHoliday || false,
        isWeekend: isWeekend || false
      },
      status: 'pending'
    });

    const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';

    try {
      const response = await axios.post(`${mlServiceUrl}/predict`, {
        forecastId: forecast._id.toString(),
        startDate,
        endDate,
        weatherConditions,
        isHoliday: isHoliday || false,
        isWeekend: isWeekend || false,
        forecastHorizon: forecastHorizon || 'all'
      });
      console.log('ML service response:', response);
      forecast.predictions = response.data.predictions;
      forecast.modelMetrics = response.data.metrics;
      forecast.status = 'completed';
      await forecast.save();

      res.status(201).json({
        status: 'success',
        data: { forecast }
      });
    } catch (mlError) {
      forecast.status = 'failed';
      await forecast.save();

      res.status(500).json({
        status: 'error',
        message: 'ML service unavailable. Using mock data for demonstration.',
        data: { forecast }
      });
    }
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
};

export const getUserForecasts = async (req, res) => {
  try {
    const forecasts = await Forecast.find({ userId: req.user._id })
      .sort({ timestamp: -1 })
      .limit(20);

    res.status(200).json({
      status: 'success',
      results: forecasts.length,
      data: { forecasts }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
};

export const getForecastById = async (req, res) => {
  try {
    const forecast = await Forecast.findOne({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!forecast) {
      return res.status(404).json({
        status: 'error',
        message: 'Forecast not found'
      });
    }

    res.status(200).json({
      status: 'success',
      data: { forecast }
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
};

export const deleteForecast = async (req, res) => {
  try {
    const forecast = await Forecast.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!forecast) {
      return res.status(404).json({
        status: 'error',
        message: 'Forecast not found'
      });
    }

    res.status(204).json({
      status: 'success',
      data: null
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message
    });
  }
};
