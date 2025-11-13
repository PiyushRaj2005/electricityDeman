import express from 'express';
import {
  createForecast,
  getUserForecasts,
  getForecastById,
  deleteForecast
} from '../controllers/forecastController.js';
import { protect } from '../controllers/authController.js';

const router = express.Router();

router.use(protect);

router.post('/', createForecast);
router.get('/', getUserForecasts);
router.get('/:id', getForecastById);
router.delete('/:id', deleteForecast);

export default router;
