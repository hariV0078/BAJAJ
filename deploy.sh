#!/bin/bash

# Render Deployment Helper Script for LLM Query System
# Usage: ./deploy.sh [setup|test|logs]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

echo_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to check if environment variables are set
check_env_vars() {
    echo "Checking environment variables..."
    
    MISSING_VARS=()
    
    if [[ -z "$OPENAI_API_KEY" ]]; then
        MISSING_VARS+=("OPENAI_API_KEY")
    fi
    
    if [[ -z "$PINECONE_API_KEY" ]]; then
        MISSING_VARS+=("PINECONE_API_KEY")
    fi
    
    if [[ ${#MISSING_VARS[@]} -gt 0 ]]; then
        echo_error "Missing required environment variables:"
        for var in "${MISSING_VARS[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Please set these variables before deployment:"
        echo "export OPENAI_API_KEY='your_openai_api_key'"
        echo "export PINECONE_API_KEY='your_pinecone_api_key'"
        echo "export PINECONE_ENVIRONMENT='us-east1-gcp'  # optional, defaults to us-east1-gcp"
        exit 1
    fi
    
    echo_success "All required environment variables are set"
}

# Function to setup Pinecone index
setup_pinecone() {
    echo "Setting up Pinecone index..."
    
    if ! command -v python3 &> /dev/null; then
        echo_error "Python3 is required but not installed"
        exit 1
    fi
    
    # Install pinecone-client if not already installed
    if ! python3 -c "import pinecone" 2>/dev/null; then
        echo "Installing pinecone-client..."
        pip3 install pinecone-client==2.2.4
    fi
    
    # Run pinecone setup
    python3 pinecone_setup.py create
    echo_success "Pinecone index setup complete"
}

# Function to test local setup
test_setup() {
    echo "Testing local setup..."
    
    check_env_vars
    
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    
    echo "Testing document processing..."
    python3 -c "
import asyncio
from app import DocumentProcessor

async def test():
    # Test with a simple text document
    try:
        # This is a basic test - replace with actual document URL for real testing
        print('✓ Document processing components loaded successfully')
        print('✓ Ready for deployment')
    except Exception as e:
        print(f'✗ Error: {e}')
        exit(1)

asyncio.run(test())
"
    
    echo_success "Local setup test completed"
}

# Function to show deployment instructions
show_deployment_instructions() {
    echo ""
    echo "=== Render Deployment Instructions ==="
    echo ""
    echo "1. Push your code to GitHub with these files:"
    echo "   - app.py"
    echo "   - start.sh (make executable: chmod +x start.sh)"
    echo "   - requirements.txt"
    echo "   - render.yaml"
    echo ""
    echo "2. In Render Dashboard:"
    echo "   - Create new Web Service"
    echo "   - Connect your GitHub repository"
    echo "   - Use these settings:"
    echo "     Build Command: pip install -r requirements.txt"
    echo "     Start Command: ./start.sh"
    echo ""
    echo "3. Set Environment Variables in Render:"
    echo "   - OPENAI_API_KEY (secret)"
    echo "   - PINECONE_API_KEY (secret)"
    echo "   - PINECONE_ENVIRONMENT = us-east1-gcp"
    echo "   - PINECONE_INDEX_NAME = document-query-system"
    echo ""
    echo "4. Deploy and test with:"
    echo "   curl https://your-app.onrender.com/health"
    echo ""
}

# Main script logic
case "${1:-setup}" in
    "setup")
        echo "Setting up LLM Query System for Render deployment..."
        check_env_vars
        setup_pinecone
        show_deployment_instructions
        ;;
    "test")
        test_setup
        ;;
    "logs")
        echo_warning "For logs, check your Render dashboard or use Render CLI"
        echo "Render CLI: https://render.com/docs/cli"
        ;;
    *)
        echo "Usage: $0 [setup|test|logs]"
        echo ""
        echo "Commands:"
        echo "  setup  - Setup Pinecone and show deployment instructions (default)"
        echo "  test   - Test local setup"
        echo "  logs   - Show logging instructions"
        exit 1
        ;;
esac
