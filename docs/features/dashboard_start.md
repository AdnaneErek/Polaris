# How to Start the Dashboard

## Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install streamlit plotly
   ```

2. **Start the dashboard**:
   ```bash
   python -m streamlit run src/dashboard.py
   ```
   
   Or use the launcher:
   ```bash
   python run_dashboard.py
   ```

3. **Access the dashboard**:
   - The dashboard will automatically open in your browser
   - If it doesn't, go to: **http://localhost:8501**

## Troubleshooting

### Dashboard won't start

**Error: "No module named streamlit"**
- Install streamlit: `pip install streamlit plotly`

**Error: "streamlit: command not found"**
- Use: `python -m streamlit run src/dashboard.py` instead of just `streamlit run`

**Port 8501 already in use**
- Stop the existing Streamlit process
- Or use a different port: `streamlit run src/dashboard.py --server.port 8502`

### Can't connect to dashboard

**Browser doesn't open automatically:**
1. Check the terminal output for the URL (usually http://localhost:8501)
2. Manually open that URL in your browser
3. Make sure no firewall is blocking port 8501

**"Connection refused" error:**
- Make sure the dashboard is actually running (check terminal)
- Try stopping and restarting: Press Ctrl+C in terminal, then run again

**Dashboard shows "No SteerCo packs found":**
- First generate a pack: `python -m src.steerco_run --as_of 2026-12-01`
- Then start the dashboard

### Charts not showing

**Error about Plotly:**
- Install plotly: `pip install plotly`
- Charts will fall back to text if Plotly isn't available

## Manual Steps

If automatic opening doesn't work:

1. **Start dashboard in terminal:**
   ```bash
   python -m streamlit run src/dashboard.py
   ```

2. **Look for this message:**
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```

3. **Copy the URL and paste it in your browser**

4. **To stop the dashboard:** Press Ctrl+C in the terminal

## Alternative: Use Specific Port

If port 8501 is busy:

```bash
python -m streamlit run src/dashboard.py --server.port 8502
```

Then access at: http://localhost:8502
