import mongoose from 'mongoose';

const forecastSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  timestamp: {
    type: Date,
    default: Date.now
  },
  inputData: {
    startDate: Date,
    endDate: Date,
    weatherConditions: {
      temperature: Number,
      humidity: Number,
      windSpeed: Number
    },
    isHoliday: Boolean,
    isWeekend: Boolean
  },
  predictions: {
    hourly: [{ time: Date, demand: Number, confidence: Number }],
    daily: [{ date: Date, demand: Number, confidence: Number }],
    weekly: [{ week: String, demand: Number, confidence: Number }],
    monthly: [{ month: String, demand: Number, confidence: Number }]
  },
  modelMetrics: {
    mse: Number,
    mae: Number,
    rmse: Number,
    r2Score: Number
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'failed'],
    default: 'pending'
  }
});

export default mongoose.model('Forecast', forecastSchema);
