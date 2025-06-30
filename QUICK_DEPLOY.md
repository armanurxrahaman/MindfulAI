# Quick Deployment Guide - MindfulAI

## 🚀 Easiest Way to Deploy (Recommended)

### Step 1: Deploy Backend on Render (Free)
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `https://github.com/armanurxrahaman/MindfulAI`
4. Configure:
   - **Name**: `mindfulai-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: Leave empty
5. Click "Create Web Service"
6. Wait for deployment (5-10 minutes)
7. Copy your backend URL (e.g., `https://mindfulai-backend.onrender.com`)

### Step 2: Update Frontend API URL
1. Open `frontend/src/App.js`
2. Find the API URL and update it to your backend URL
3. Commit and push changes

### Step 3: Deploy Frontend on Vercel (Free)
1. Go to [vercel.com](https://vercel.com) and sign up
2. Click "New Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: `Create React App`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
5. Click "Deploy"
6. Your site will be live in 2-3 minutes!

## 🎯 Alternative: All-in-One Vercel Deployment

If you want everything on Vercel:
1. The `vercel.json` file is already configured
2. Deploy directly to Vercel
3. Both frontend and backend will be on the same domain

## 🔧 Environment Variables (if needed)

Add these to your deployment platform:
```
CORS_ORIGINS=https://your-frontend-domain.vercel.app
```

## 📱 Your Live URLs
- **Frontend**: `https://your-project.vercel.app`
- **Backend API**: `https://your-backend.onrender.com`
- **API Docs**: `https://your-backend.onrender.com/docs`

## ✅ What You'll Get
- ✅ Live website accessible from anywhere
- ✅ Mobile-responsive design
- ✅ Fast API endpoints
- ✅ Automatic HTTPS
- ✅ Free hosting (with limits)

## 🚨 Important Notes
- Model files are excluded by `.gitignore` (too large for free hosting)
- You may need to adjust model loading for production
- Consider using cloud storage for large files

---

**Need help?** Check the main README.md for detailed setup instructions. 