# UPIR Web Application

A Google App Engine web application for the Universal Plan Intermediate Representation (UPIR) system.

## Features

- **Interactive Demo**: Try UPIR's code generation capabilities in real-time
- **Multi-language Support**: Generate code in Python, Go, and JavaScript
- **Live Verification**: Verify specifications and get immediate feedback
- **Parameter Synthesis**: Use Z3 to synthesize optimal parameters
- **Documentation**: Complete documentation and research paper access
- **Responsive Design**: Works on desktop and mobile devices

## Local Development

### Prerequisites

- Python 3.9+
- Google Cloud SDK (for deployment)

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
```bash
python main.py
```

The application will be available at http://localhost:8080

## Deployment to Google App Engine

### Prerequisites

1. Install Google Cloud SDK:
```bash
# macOS
brew install google-cloud-sdk

# Or download from https://cloud.google.com/sdk/docs/install
```

2. Authenticate:
```bash
gcloud auth login
```

3. Set project:
```bash
gcloud config set project subhadipmitra-pso-team-369906
```

### Deploy

Use the deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

Or deploy manually:
```bash
gcloud app deploy app.yaml --project subhadipmitra-pso-team-369906
```

The application will be available at:
https://subhadipmitra-pso-team-369906.appspot.com

## Project Structure

```
webapp/
├── app.yaml              # App Engine configuration
├── main.py               # Flask application
├── requirements.txt      # Python dependencies
├── deploy.sh            # Deployment script
├── templates/           # HTML templates
│   ├── base.html       # Base template
│   ├── index.html      # Landing page
│   └── demo.html       # Interactive demo
├── static/             # Static assets
│   ├── css/           # Stylesheets
│   │   └── style.css  # Main styles
│   └── js/            # JavaScript
│       └── main.js    # Main scripts
└── README.md          # This file
```

## API Endpoints

### POST /api/generate
Generate code from UPIR template.

Request:
```json
{
  "template": "queue_worker",
  "language": "python"
}
```

Response:
```json
{
  "success": true,
  "code": "...",
  "language": "python",
  "template": "Queue Worker",
  "generation_time": 2.83,
  "parameters": {...}
}
```

### POST /api/verify
Verify UPIR specification.

Request:
```json
{
  "specification": "system Test { ... }"
}
```

Response:
```json
{
  "success": true,
  "valid": true,
  "errors": [],
  "warnings": [],
  "verification_time": 14.0,
  "properties_verified": 12
}
```

### POST /api/synthesize
Synthesize optimal parameters using Z3.

Request:
```json
{
  "constraints": "throughput >= 5000\nlatency <= 100"
}
```

Response:
```json
{
  "success": true,
  "parameters": {
    "batch_size": 94,
    "workers": 14,
    "throughput": 13160,
    "synthesis_time": 114.1
  },
  "satisfiable": true
}
```

## Configuration

Edit `app.yaml` to configure:
- Scaling parameters
- Environment variables
- Static file handlers

## Development Tips

1. **Hot Reload**: The Flask development server supports hot reload
2. **Debugging**: Set `debug=True` in `app.run()` for development
3. **Testing**: Test locally before deploying to App Engine
4. **Logs**: View logs with `gcloud app logs tail -s default`

## Security Considerations

- Change the `SECRET_KEY` in production
- Enable HTTPS (already configured in app.yaml)
- Add authentication if needed
- Validate all user inputs
- Rate limit API endpoints in production

## License

Copyright 2025 Google Cloud Professional Services

## Support

For issues or questions about UPIR, please refer to the main project documentation.