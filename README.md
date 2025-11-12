# CSSM Electricity Demand Forecasting System

A comprehensive deep learning application for multi-horizon electricity demand forecasting using Cross-Scale Sequence Mixer (CSSM) architecture.

## Overview

This application implements a novel deep learning architecture that captures multi-resolution temporal dependencies by learning from hourly, daily, weekly, and monthly patterns simultaneously. The system uses LSTM-based encoders for each time scale and fuses their outputs using a cross-scale attention mechanism.

## Dataset

This project uses the **Delhi 5-Minute Electricity Demand** dataset from Kaggle:
- **Dataset Link**: https://www.kaggle.com/datasets/vinayaktrivedi/delhi-5-minute-electricity-demand
- **Description**: Historical electricity demand data for Delhi with weather information
- **Features**: Timestamp, demand, temperature, humidity, wind speed

## Architecture

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Axios** for API communication

### Backend (Node.js)
- **Express.js** REST API
- **MongoDB** with Mongoose ODM
- **JWT** authentication
- **bcrypt** for password hashing

### ML Service (Python)
- **PyTorch** for deep learning
- **Flask** for API endpoints
- **CSSM Model**: LSTM encoders with cross-attention
- Multi-scale forecasting (hourly, daily, weekly, monthly)

## Project Structure

```
.
├── src/                          # React frontend
│   ├── components/              # React components
│   │   ├── Login.tsx
│   │   ├── Signup.tsx
│   │   ├── Dashboard.tsx
│   │   ├── ForecastForm.tsx
│   │   ├── ForecastResults.tsx
│   │   └── ForecastHistory.tsx
│   ├── context/                 # React context
│   │   └── AuthContext.tsx
│   ├── services/                # API services
│   │   └── api.ts
│   ├── types/                   # TypeScript types
│   │   └── index.ts
│   └── App.tsx
│
├── backend/                      # Node.js backend
│   ├── models/                  # MongoDB models
│   │   ├── User.js
│   │   └── Forecast.js
│   ├── controllers/             # Request handlers
│   │   ├── authController.js
│   │   └── forecastController.js
│   ├── routes/                  # API routes
│   │   ├── authRoutes.js
│   │   └── forecastRoutes.js
│   ├── ml_service/              # Python ML service
│   │   ├── cssm_model.py       # CSSM architecture
│   │   ├── data_preprocessor.py
│   │   ├── app.py              # Flask API
│   │   └── requirements.txt
│   ├── server.js
│   ├── package.json
│   └── .env
│
└── package.json                  # Frontend dependencies
```

## Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- MongoDB 6.0+

### 1. Install Frontend Dependencies
```bash
npm install
```

### 2. Install Backend Dependencies
```bash
cd backend
npm install
```

### 3. Install Python ML Service
```bash
cd backend/ml_service
pip install -r requirements.txt
```

### 4. Setup MongoDB
```bash
# Install MongoDB (Ubuntu/Debian)
sudo apt-get install mongodb

# Start MongoDB service
sudo systemctl start mongodb

# Verify MongoDB is running
sudo systemctl status mongodb
```

## Configuration

### Backend Configuration
Edit `backend/.env`:
```env
MONGODB_URI=mongodb://localhost:27017/cssm_forecasting
JWT_SECRET=your_secure_jwt_secret_key
PORT=5000
ML_SERVICE_URL=http://localhost:8000
```

## Running the Application

### 1. Start MongoDB
```bash
sudo systemctl start mongodb
```

### 2. Start Backend Server
```bash
cd backend
npm start
```
Backend runs on: http://localhost:5000

### 3. Start ML Service
```bash
cd backend/ml_service
python app.py
```
ML Service runs on: http://localhost:8000

### 4. Start Frontend
```bash
npm run dev
```
Frontend runs on: http://localhost:5173

## Usage

### 1. User Authentication
- Navigate to http://localhost:5173
- Create an account using the signup form
- Login with your credentials

### 2. Generate Forecasts
- Enter weather parameters:
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (km/h)
- Select special conditions:
  - Holiday checkbox
  - Weekend checkbox
- Click "Generate Forecast"

### 3. View Results
- Switch between time scales:
  - **Hourly**: Next 24 hours
  - **Daily**: Next 7 days
  - **Weekly**: Next 4 weeks
  - **Monthly**: Next 12 months
- View model performance metrics:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score

### 4. Forecast History
- Click "History" tab to view past forecasts
- Click any forecast to view details

## CSSM Model Architecture

### Key Components

1. **Multi-Scale Encoders**
   - Hourly Encoder: Captures short-term patterns
   - Daily Encoder: Captures daily cycles
   - Weekly Encoder: Captures weekly patterns
   - Monthly Encoder: Captures seasonal trends

2. **Cross-Scale Attention**
   - Multi-head attention mechanism
   - Dynamically weights information from different scales
   - Enables the model to focus on relevant temporal patterns

3. **Fusion Layer**
   - Combines multi-scale representations
   - Dense neural network with dropout
   - Produces unified feature representation

4. **Multi-Horizon Predictors**
   - Separate prediction heads for each time scale
   - Outputs forecasts for multiple horizons simultaneously

### Model Parameters
- Input Size: 10 features
- Hidden Size: 128 units
- Number of Scales: 4 (hourly, daily, weekly, monthly)
- Attention Heads: 4

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Create new account
- `POST /api/auth/login` - User login

### Forecasts
- `POST /api/forecasts` - Create new forecast
- `GET /api/forecasts` - Get user forecasts
- `GET /api/forecasts/:id` - Get specific forecast
- `DELETE /api/forecasts/:id` - Delete forecast

### ML Service
- `GET /health` - Health check
- `POST /predict` - Generate predictions
- `GET /model/info` - Model information

## Features

- User authentication and authorization
- Real-time electricity demand forecasting
- Multi-scale predictions (hourly to monthly)
- Interactive visualizations with Recharts
- Model performance metrics
- Forecast history management
- Weather condition integration
- Holiday and weekend adjustments

## Performance Metrics

The model provides comprehensive evaluation metrics:
- **MSE**: Measures average squared difference
- **MAE**: Average absolute error
- **RMSE**: Square root of MSE
- **R² Score**: Coefficient of determination (model accuracy)

## Technologies Used

### Frontend
- React 18.3.1
- TypeScript 5.5.3
- Tailwind CSS 3.4.1
- Recharts 3.2.1
- Axios 1.12.2
- Lucide React (icons)

### Backend
- Express.js 4.18.2
- MongoDB with Mongoose 8.0.0
- JWT for authentication
- bcryptjs for password hashing

### ML Service
- PyTorch 2.1.0
- Flask 3.0.0
- NumPy 1.24.3
- Pandas 2.1.1
- Scikit-learn 1.3.1

## Development

### Build Frontend
```bash
npm run build
```

### Lint Code
```bash
npm run lint
```

### Type Check
```bash
npm run typecheck
```

## Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
sudo systemctl status mongodb

# Restart MongoDB
sudo systemctl restart mongodb
```

### Backend Not Starting
- Verify MongoDB is running
- Check port 5000 is available
- Review backend/.env configuration

### ML Service Errors
- Install all Python dependencies
- Check Python version (3.8+)
- Verify PyTorch installation

## Future Enhancements

- Real-time data ingestion from Kaggle dataset
- Model training interface
- Advanced hyperparameter tuning
- Multiple model comparison
- Export forecasts to CSV/PDF
- Email notifications for forecast completion
- Real-time dashboard updates

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository.
