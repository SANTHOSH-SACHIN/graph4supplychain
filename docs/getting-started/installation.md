# Installation Guide

This guide will help you set up Graph4SupplyChain on your system. You can choose between Docker-based installation or manual setup.

## Docker Installation (Recommended)

The easiest way to get started is using Docker:

1. Clone the repository:
   ```bash
   git clone https://github.com/SANTHOSH-SACHIN/graph4supplychain.git
   cd graph4supplychain
   ```

2. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

The application will be available at `http://localhost:8501`.

## Manual Installation

If you prefer to run the application directly on your system:

1. Clone the repository:
   ```bash
   git clone https://github.com/SANTHOSH-SACHIN/graph4supplychain.git
   cd graph4supplychain
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.default .env
   # Edit .env file with your configurations
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## System Requirements

- Python 3.8 or higher
- Docker and Docker Compose (for Docker installation)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Dependencies

Key dependencies include:

- Streamlit
- PyTorch
- Prophet
- XGBoost
- Statsmodels
- Pandas
- NetworkX

For a complete list, see `requirements.txt`.

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - If port 8501 is already in use, you can change it in `docker-compose.yml` or use:
     ```bash
     streamlit run app.py --server.port [PORT_NUMBER]
     ```

2. **Memory Issues**
   - Adjust Docker memory limits in Docker Desktop settings
   - For manual installation, ensure sufficient system memory

3. **Package Conflicts**
   - Try creating a fresh virtual environment
   - Update pip: `pip install --upgrade pip`
   - Install packages one by one if conflicts occur

