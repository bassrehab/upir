#!/bin/bash

# UPIR Web Application Deployment Script for Google App Engine

echo "========================================="
echo "UPIR Web Application Deployment"
echo "========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
PROJECT_ID="subhadipmitra-pso-team-369906"
echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable appengine.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Initialize App Engine if needed
echo "Checking App Engine..."
gcloud app describe &> /dev/null || gcloud app create --region=us-central

# Deploy the application
echo "Deploying UPIR web application..."
gcloud app deploy app.yaml --quiet

# Deploy cron jobs if exists
if [ -f "cron.yaml" ]; then
    echo "Deploying cron jobs..."
    gcloud app deploy cron.yaml --quiet
fi

# Deploy dispatch rules if exists
if [ -f "dispatch.yaml" ]; then
    echo "Deploying dispatch rules..."
    gcloud app deploy dispatch.yaml --quiet
fi

echo "========================================="
echo "Deployment complete!"
echo "Application URL: https://$PROJECT_ID.appspot.com"
echo "========================================="

# Open the app in browser
read -p "Open application in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud app browse
fi