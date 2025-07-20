#!/bin/bash

# GitHub Secrets Setup Script for SF Business Model CI/CD
# Run this script to automatically add Docker Hub credentials to GitHub

set -e

echo "🔐 Setting up GitHub Secrets for CI/CD Pipeline"
echo "=============================================="

# Repository details
REPO="esengendo/San-Francisco-Business-Model"
DOCKER_USERNAME="esengendo730"
DOCKER_TOKEN="${DOCKER_TOKEN:-}"

# Check if Docker token is provided
if [ -z "$DOCKER_TOKEN" ]; then
    echo "❌ DOCKER_TOKEN environment variable not set."
    echo "📝 Usage: DOCKER_TOKEN=your_token ./setup_github_secrets.sh"
    echo "🔑 Or set it manually: export DOCKER_TOKEN=your_token"
    exit 1
fi

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "📥 Install it from: https://cli.github.com/"
    echo ""
    echo "🔧 Alternative: Add secrets manually at:"
    echo "   https://github.com/$REPO/settings/secrets/actions"
    echo ""
    echo "   DOCKER_USERNAME: $DOCKER_USERNAME"
    echo "   DOCKER_PASSWORD: [your Docker Hub token]"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &>/dev/null; then
    echo "🔑 Please authenticate with GitHub CLI first:"
    echo "   gh auth login"
    exit 1
fi

echo "✅ GitHub CLI detected and authenticated"
echo ""

# Add secrets
echo "📝 Adding DOCKER_USERNAME secret..."
echo "$DOCKER_USERNAME" | gh secret set DOCKER_USERNAME --repo="$REPO"

echo "🔑 Adding DOCKER_PASSWORD secret..."
echo "$DOCKER_TOKEN" | gh secret set DOCKER_PASSWORD --repo="$REPO"

echo ""
echo "✅ GitHub Secrets successfully added!"
echo ""
echo "🚀 Next steps:"
echo "1. Visit: https://github.com/$REPO/actions"
echo "2. The CI/CD pipeline will trigger automatically on next push"
echo "3. Monitor the first workflow run for successful Docker deployment"
echo ""
echo "🐳 Docker Hub: https://hub.docker.com/r/$DOCKER_USERNAME/sanfrancisco_business_model"
echo "📊 GitHub Actions: https://github.com/$REPO/actions"
echo ""
echo "🎉 Your repository now has automated CI/CD with Docker Hub integration!"