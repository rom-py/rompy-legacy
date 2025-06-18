#!/bin/bash

# Setup Script for Repository Splitting Tools
# ===========================================
#
# This script sets up all the necessary tools and dependencies for splitting
# the rompy repository into multiple focused repositories while preserving
# git history, branches, and tags.
#
# Usage:
#   chmod +x setup_split_tools.sh
#   ./setup_split_tools.sh
#
# Or with options:
#   ./setup_split_tools.sh --system-install  # Install system-wide
#   ./setup_split_tools.sh --check-only      # Only check prerequisites

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.7"
GIT_MIN_VERSION="2.20"
VENV_NAME="rompy-split-env"

# Parse command line arguments
SYSTEM_INSTALL=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --system-install)
            SYSTEM_INSTALL=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --system-install    Install packages system-wide instead of in venv"
            echo "  --check-only        Only check prerequisites, don't install anything"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Utility functions
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Version comparison function
version_compare() {
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    print_header "Checking Python Installation"

    local python_cmd=""

    # Try different Python commands
    for cmd in python3 python; do
        if command_exists "$cmd"; then
            local version
            version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            if [[ -n $version ]]; then
                version_compare "$version" "$PYTHON_MIN_VERSION"
                local result=$?
                if [[ $result -eq 0 ]] || [[ $result -eq 1 ]]; then
                    python_cmd="$cmd"
                    print_success "Found Python $version at $(which $cmd)"
                    break
                else
                    print_warning "Found Python $version at $(which $cmd), but minimum version $PYTHON_MIN_VERSION is required"
                fi
            fi
        fi
    done

    if [[ -z $python_cmd ]]; then
        print_error "Python $PYTHON_MIN_VERSION or higher is required but not found"
        print_info "Please install Python from https://python.org or your package manager"
        return 1
    fi

    export PYTHON_CMD="$python_cmd"
    return 0
}

# Check Git version
check_git() {
    print_header "Checking Git Installation"

    if ! command_exists git; then
        print_error "Git is not installed"
        print_info "Please install Git from https://git-scm.com or your package manager"
        return 1
    fi

    local git_version
    git_version=$(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

    if [[ -z $git_version ]]; then
        print_error "Could not determine Git version"
        return 1
    fi

    version_compare "$git_version" "$GIT_MIN_VERSION"
    local result=$?

    if [[ $result -eq 2 ]]; then
        print_error "Git version $git_version found, but minimum version $GIT_MIN_VERSION is required"
        print_info "Please update Git to a newer version"
        return 1
    fi

    print_success "Found Git $git_version at $(which git)"
    return 0
}

# Check if we're in a git repository
check_git_repo() {
    print_header "Checking Git Repository"

    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        print_error "Not in a Git repository"
        print_info "Please run this script from within your rompy git repository"
        return 1
    fi

    local repo_root
    repo_root=$(git rev-parse --show-toplevel)
    print_success "Found Git repository at: $repo_root"

    # Check if rompy directory exists
    if [[ ! -d "rompy" ]]; then
        print_warning "rompy directory not found in current location"
        print_info "Make sure you're in the root of the rompy repository"
    else
        print_success "Found rompy source directory"
    fi

    return 0
}

# Create virtual environment
create_venv() {
    print_header "Setting Up Python Virtual Environment"

    if [[ $SYSTEM_INSTALL == true ]]; then
        print_info "Skipping virtual environment creation (system install requested)"
        return 0
    fi

    if [[ -d "$VENV_NAME" ]]; then
        print_info "Virtual environment '$VENV_NAME' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            print_info "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
        fi
    fi

    if [[ ! -d "$VENV_NAME" ]]; then
        print_info "Creating virtual environment: $VENV_NAME"
        "$PYTHON_CMD" -m venv "$VENV_NAME"
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    # shellcheck source=/dev/null
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    return 0
}

# Install Python packages
install_packages() {
    print_header "Installing Python Packages"

    if [[ $SYSTEM_INSTALL == false ]] && [[ ! -d "$VENV_NAME" ]]; then
        print_error "Virtual environment not found. Run without --check-only first."
        return 1
    fi

    # Activate virtual environment if not system install
    if [[ $SYSTEM_INSTALL == false ]]; then
        # shellcheck source=/dev/null
        source "$VENV_NAME/bin/activate"
    fi

    # Check if requirements file exists
    if [[ ! -f "split_requirements.txt" ]]; then
        print_error "split_requirements.txt not found"
        print_info "This file should be in the same directory as this script"
        return 1
    fi

    print_info "Installing packages from split_requirements.txt..."
    pip install -r split_requirements.txt

    print_success "All Python packages installed successfully"
    return 0
}

# Validate installation
validate_installation() {
    print_header "Validating Installation"

    # Activate virtual environment if not system install
    if [[ $SYSTEM_INSTALL == false ]] && [[ -d "$VENV_NAME" ]]; then
        # shellcheck source=/dev/null
        source "$VENV_NAME/bin/activate"
    fi

    local all_good=true

    # Check git-filter-repo
    if command_exists git-filter-repo; then
        local version
        version=$(git-filter-repo --version 2>&1 | head -1)
        print_success "git-filter-repo is available: $version"
    else
        print_error "git-filter-repo is not available"
        all_good=false
    fi

    # Check PyYAML
    if python -c "import yaml; print('PyYAML version:', yaml.__version__)" 2>/dev/null; then
        print_success "PyYAML is available"
    else
        print_error "PyYAML is not available"
        all_good=false
    fi

    # Check for optional dependencies
    if python -c "import tomli, tomli_w" 2>/dev/null; then
        print_success "TOML libraries (tomli, tomli_w) are available"
    else
        print_warning "TOML libraries not available (optional, for pyproject.toml updates)"
    fi

    # Check configuration files
    if [[ -f "repo_split_config.yaml" ]]; then
        print_success "Configuration file found: repo_split_config.yaml"
    else
        print_warning "Configuration file not found: repo_split_config.yaml"
    fi

    # Check scripts
    local scripts=("split_repository.py" "validate_config.py")
    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            print_success "Script found: $script"
        else
            print_error "Script not found: $script"
            all_good=false
        fi
    done

    if [[ $all_good == true ]]; then
        print_success "All components are properly installed and available!"
        return 0
    else
        print_error "Some components are missing or not working properly"
        return 1
    fi
}

# Show usage instructions
show_usage() {
    print_header "Usage Instructions"

    cat << EOF

Your repository splitting tools are now set up! Here's how to use them:

${GREEN}1. Validate Configuration:${NC}
   python validate_config.py --config repo_split_config.yaml

${GREEN}2. Run Dry Run:${NC}
   python split_repository.py --config repo_split_config.yaml --dry-run

${GREEN}3. Execute Split:${NC}
   python split_repository.py --config repo_split_config.yaml

${GREEN}4. Using Makefile (if available):${NC}
   make -f Makefile.split help
   make -f Makefile.split validate
   make -f Makefile.split dry-run
   make -f Makefile.split split

${YELLOW}Virtual Environment:${NC}
EOF

    if [[ $SYSTEM_INSTALL == false ]]; then
        cat << EOF
   To activate: source $VENV_NAME/bin/activate
   To deactivate: deactivate
EOF
    else
        echo "   Packages installed system-wide"
    fi

    cat << EOF

${BLUE}Configuration File:${NC} repo_split_config.yaml
${BLUE}Documentation:${NC} REPOSITORY_SPLITTING.md

${YELLOW}Before running the split:${NC}
- Review and customize repo_split_config.yaml
- Commit any uncommitted changes
- Consider creating a backup of your repository

For detailed instructions, see REPOSITORY_SPLITTING.md
EOF
}

# Main execution
main() {
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                Repository Splitting Setup Tool              ║
║                                                              ║
║  This script will set up all tools needed to split your     ║
║  rompy repository into multiple focused repositories while   ║
║  preserving git history, branches, and tags.                ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    # Prerequisites check
    if ! check_python; then
        exit 1
    fi

    if ! check_git; then
        exit 1
    fi

    if ! check_git_repo; then
        exit 1
    fi

    # If only checking, stop here
    if [[ $CHECK_ONLY == true ]]; then
        print_header "Prerequisites Check Complete"
        print_success "All prerequisites are satisfied!"
        exit 0
    fi

    # Installation steps
    if ! create_venv; then
        exit 1
    fi

    if ! install_packages; then
        exit 1
    fi

    if ! validate_installation; then
        exit 1
    fi

    # Success!
    show_usage

    print_header "Setup Complete!"
    print_success "Repository splitting tools are ready to use!"

    if [[ $SYSTEM_INSTALL == false ]]; then
        echo ""
        print_info "Don't forget to activate the virtual environment:"
        echo -e "  ${GREEN}source $VENV_NAME/bin/activate${NC}"
    fi
}

# Run main function
main "$@"
