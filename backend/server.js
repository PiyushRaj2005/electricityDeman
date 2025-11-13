import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import authRoutes from './routes/authRoutes.js';
import forecastRoutes from './routes/forecastRoutes.js';

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api/auth', authRoutes);
app.use('/api/forecasts', forecastRoutes);

app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'success',
    message: 'CSSM Backend API is running',
    timestamp: new Date().toISOString()
  });
});

const PORT = process.env.PORT || 5000;
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://ishaan:ishaan@cluster0.12fuxtv.mongodb.net/';

mongoose
  .connect(MONGODB_URI)
  .then(() => {
    console.log('MongoDB connected successfully');
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      console.log(`MongoDB URI: ${MONGODB_URI}`);
    });
  })
  .catch((error) => {
    console.error('MongoDB connection error:', error);
    console.log('Server starting without database connection...');
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT} (without database)`);
    });
  });

export default app;
