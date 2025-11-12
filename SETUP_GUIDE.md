# CSSM Application Setup Guide

This guide provides step-by-step instructions to set up and run the CSSM Electricity Demand Forecasting application.

## Prerequisites

Before starting, ensure you have the following installed:

- **Node.js** 18+ and npm
- **Python** 3.8+
- **MongoDB** 6.0+

## Quick Start

### Step 1: Install MongoDB

#### Ubuntu/Debian:
```bash
# Import MongoDB public GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Update package list
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod

# Enable MongoDB to start on boot
sudo systemctl enable mongod

# Verify MongoDB is running
sudo systemctl status mongod
```

#### MacOS:
```bash
# Install MongoDB using Homebrew
brew tap mongodb/brew
brew install mongodb-community@6.0

# Start MongoDB service
brew services start mongodb-community@6.0

# Verify installation
mongosh --version
```

#### Windows:
1. Download MongoDB Community Server from https://www.mongodb.com/try/download/community
2. Run the installer
3. Choose "Complete" installation
4. Install as a service
5. Verify installation: Open Command Prompt and run `mongod --version`

### Step 2: Install Frontend Dependencies

```bash
# From project root directory
npm install
```

### Step 3: Install Backend Dependencies

```bash
# Navigate to backend directory
cd backend

# Install Node.js dependencies
npm install

# Go back to root
cd ..
```

### Step 4: Install Python ML Service Dependencies

```bash
# Navigate to ML service directory
cd backend/ml_service

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Go back to root
cd ../..
```

### Step 5: Configure Environment Variables

The backend environment file is already configured at `backend/.env`:

```env
MONGODB_URI=mongodb://localhost:27017/cssm_forecasting
JWT_SECRET=your_jwt_secret_key_change_in_production
PORT=5000
ML_SERVICE_URL=http://localhost:8000
```

**Important**: Change `JWT_SECRET` for production deployment!

### Step 6: Start All Services

Open **three separate terminal windows**:

#### Terminal 1 - MongoDB (if not running as service)
```bash
mongod
```

#### Terminal 2 - Backend Server
```bash
cd backend
npm start
```

You should see:
```
MongoDB connected successfully
Server running on port 5000
```

#### Terminal 3 - ML Service
```bash
cd backend/ml_service

# Activate virtual environment if you created one
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

python app.py
```

You should see:
```
CSSM Model initialized successfully
 * Running on http://0.0.0.0:8000
```

#### Terminal 4 - Frontend (Development Mode)
```bash
# From project root
npm run dev
```

You should see:
```
VITE v5.4.8  ready in XXX ms
➜  Local:   http://localhost:5173/
```

### Step 7: Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

## First Time Usage

### 1. Create an Account
- Click on "Sign up" button
- Enter your full name, email, and password
- Click "Create Account"

### 2. Generate Your First Forecast
- After logging in, you'll see the dashboard
- Enter weather parameters:
  - Temperature (e.g., 25°C)
  - Humidity (e.g., 60%)
  - Wind Speed (e.g., 10 km/h)
- Check "Holiday" or "Weekend" if applicable
- Click "Generate Forecast"

### 3. View Results
- The forecast will appear below the form
- Switch between time scales: Hourly, Daily, Weekly, Monthly
- View model performance metrics at the bottom

### 4. View History
- Click "History" tab to see all your forecasts
- Click any forecast to view its details

## Troubleshooting

### MongoDB Connection Issues

**Problem**: Backend shows "MongoDB connection error"

**Solution**:
```bash
# Check if MongoDB is running
sudo systemctl status mongod

# If not running, start it
sudo systemctl start mongod

# Check logs
sudo tail -f /var/log/mongodb/mongod.log
```

### Port Already in Use

**Problem**: "Error: listen EADDRINUSE: address already in use :::5000"

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process (replace PID with actual process ID)
kill -9 PID

# Or use a different port in backend/.env
PORT=5001
```

### Python Dependencies Issues

**Problem**: "ModuleNotFoundError: No module named 'torch'"

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# If torch installation fails, try:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Frontend Not Starting

**Problem**: "Error: Cannot find module..."

**Solution**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### CORS Errors

**Problem**: "Access to XMLHttpRequest blocked by CORS policy"

**Solution**:
- Verify backend is running on port 5000
- Check that `backend/server.js` has CORS enabled (it should be by default)
- Clear browser cache and reload

## Production Deployment

### Building Frontend for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Running Backend in Production

```bash
cd backend

# Install PM2 process manager
npm install -g pm2

# Start backend with PM2
pm2 start server.js --name cssm-backend

# Start ML service with PM2
cd ml_service
pm2 start app.py --name cssm-ml --interpreter python3

# Save PM2 configuration
pm2 save

# Enable PM2 to start on boot
pm2 startup
```

### Serving Frontend with Nginx

Install Nginx:
```bash
sudo apt-get install nginx
```

Configure Nginx (`/etc/nginx/sites-available/cssm`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/project/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/cssm /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Dataset Integration

To use the actual Delhi electricity demand dataset:

1. **Download Dataset**:
   - Visit: https://www.kaggle.com/datasets/vinayaktrivedi/delhi-5-minute-electricity-demand
   - Download the CSV file

2. **Prepare Data**:
   ```bash
   cd backend/ml_service
   mkdir data
   # Copy downloaded CSV to data/
   ```

3. **Process Data**:
   Create a script to load and process the CSV:
   ```python
   import pandas as pd

   # Load dataset
   df = pd.read_csv('data/delhi_electricity_demand.csv')

   # Preprocess and aggregate to hourly
   # ... (add preprocessing code)

   # Save processed data
   df.to_csv('data/processed_data.csv', index=False)
   ```

4. **Train Model**:
   Update `cssm_model.py` to load actual data and train the model

## Monitoring

### Check Service Status

```bash
# Backend
curl http://localhost:5000/api/health

# ML Service
curl http://localhost:8000/health
```

### View Logs

```bash
# Backend logs (if using PM2)
pm2 logs cssm-backend

# ML Service logs
pm2 logs cssm-ml

# MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log
```

## Security Recommendations

1. **Change Default Secrets**:
   - Update `JWT_SECRET` in `backend/.env`
   - Use strong, random values in production

2. **Enable MongoDB Authentication**:
   ```bash
   # Connect to MongoDB
   mongosh

   # Create admin user
   use admin
   db.createUser({
     user: "admin",
     pwd: "secure_password",
     roles: ["userAdminAnyDatabase"]
   })

   # Update backend/.env
   MONGODB_URI=mongodb://admin:secure_password@localhost:27017/cssm_forecasting?authSource=admin
   ```

3. **Use HTTPS**:
   - Obtain SSL certificate (Let's Encrypt)
   - Configure Nginx with SSL

4. **Firewall Configuration**:
   ```bash
   # Allow only necessary ports
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

## Support

For issues or questions:
1. Check this setup guide
2. Review the main README.md
3. Check application logs
4. Consult the research report for technical details

## Next Steps

After successful setup:
1. Explore the dashboard features
2. Generate multiple forecasts with different parameters
3. Review the research report for understanding the model
4. Customize the application for your specific needs

---

**Happy Forecasting!**
