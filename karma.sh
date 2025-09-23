#!/bin/bash

# KARMA Pipeline Execution Script
# This script provides an easy interface to run the KARMA pipeline

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
PURPLE='\\033[0;35m'
CYAN='\\033[0;36m'
NC='\\033[0m' # No Color

# Default values
API_KEY="${KARMA_API_KEY}"
MODEL="gpt-4o-mini"
RELEVANCE_THRESHOLD="0.2"
INTEGRATION_THRESHOLD="0.5"
OUTPUT_DIR="karma_output"
DOMAIN="biomedical"
VERBOSE=""

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}    KARMA Pipeline Runner${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <input_file>"
    echo ""
    echo "Options:"
    echo "  -k, --api-key KEY           OpenAI API key (default: from KARMA_API_KEY env)"
    echo "  -m, --model MODEL           Model to use (default: gpt-4o-mini)"
    echo "  -r, --relevance-threshold   Relevance threshold (default: 0.2)"
    echo "  -i, --integration-threshold Integration threshold (default: 0.5)"
    echo "  -o, --output-dir DIR        Output directory (default: karma_output)"
    echo "  -d, --domain DOMAIN         Domain context (default: biomedical)"
    echo "  -v, --verbose               Verbose output"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 document.pdf"
    echo "  $0 -m gpt-4 -r 0.3 -i 0.7 document.pdf"
    echo "  $0 --verbose --output-dir results/ document.pdf"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found. Please install Python 3.8+"
        exit 1
    fi

    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "Python 3.8+ is required. Found: $python_version"
        exit 1
    fi

    # Check if main.py exists
    if [[ ! -f "main.py" ]]; then
        print_error "main.py not found. Please run this script from the KARMA repository root."
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to run the pipeline
run_pipeline() {
    local input_file="$1"

    print_header
    print_info "Starting KARMA pipeline..."
    print_info "Input: $input_file"
    print_info "Model: $MODEL"
    print_info "Output: $OUTPUT_DIR"

    # Check if input file exists
    if [[ ! -f "$input_file" ]]; then
        print_error "Input file not found: $input_file"
        exit 1
    fi

    # Build command
    cmd="python3 main.py"
    cmd="$cmd \\"$input_file\\""
    cmd="$cmd --api-key \\"$API_KEY\\""
    cmd="$cmd --model $MODEL"
    cmd="$cmd --relevance-threshold $RELEVANCE_THRESHOLD"
    cmd="$cmd --integration-threshold $INTEGRATION_THRESHOLD"
    cmd="$cmd --output-dir \\"$OUTPUT_DIR\\""
    cmd="$cmd --domain $DOMAIN"

    if [[ -n "$VERBOSE" ]]; then
        cmd="$cmd --verbose"
    fi

    print_info "Executing: $cmd"
    echo ""

    # Run the command
    eval $cmd
    exit_code=$?

    echo ""
    if [[ $exit_code -eq 0 ]]; then
        print_success "Pipeline completed successfully!"
        print_info "Results saved to: $OUTPUT_DIR"
    else
        print_error "Pipeline failed with exit code $exit_code"
        exit $exit_code
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--relevance-threshold)
            RELEVANCE_THRESHOLD="$2"
            shift 2
            ;;
        -i|--integration-threshold)
            INTEGRATION_THRESHOLD="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # This should be the input file
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

# Check if input file was provided
if [[ -z "$INPUT_FILE" ]]; then
    print_error "Input file is required"
    show_usage
    exit 1
fi

# Check if API key is available
if [[ -z "$API_KEY" ]]; then
    print_error "API key is required. Set KARMA_API_KEY environment variable or use -k option"
    exit 1
fi

# Run prerequisite checks
check_prerequisites

# Run the pipeline
run_pipeline "$INPUT_FILE"