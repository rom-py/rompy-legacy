#!/bin/bash
#
# SCHISM Boundary Conditions Examples Test Runner
#
# This script runs through all the boundary condition configuration examples
# sequentially to validate their functionality and demonstrate their usage.
#
# Usage: ./run_boundary_conditions_examples.sh [OPTIONS]
#
# Options:
#   --all              Run all examples (default)
#   --tidal            Run only tidal examples
#   --hybrid           Run only hybrid examples
#   --river            Run only river examples
#   --nested           Run only nested examples
#   --single <name>    Run single example by name
#   --dry-run          Show what would be run without executing
#   --keep-outputs     Keep output directories after run
#   --help             Show this help message

set -e

# Function to find project root
find_project_root() {
    local current_dir="$PWD"

    # Look for common project root indicators
    while [[ "$current_dir" != "/" ]]; do
        if [[ -f "$current_dir/setup.py" ]] || [[ -f "$current_dir/pyproject.toml" ]] || [[ -d "$current_dir/.git" ]]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done

    # If not found, assume current directory
    echo "$PWD"
}

# Configuration
SCHISM_VERSION="v5.11.1"
PROJECT_ROOT="$(find_project_root)"
BASE_OUTPUT_DIR="$PROJECT_ROOT/boundary_conditions_test_outputs"
EXAMPLES_DIR="$PROJECT_ROOT/notebooks/schism/boundary_conditions_examples"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Example definitions with metadata
declare -A EXAMPLES
declare -A EXAMPLE_DESCRIPTIONS
declare -A EXAMPLE_CATEGORIES

# Tidal-only examples
EXAMPLES["basic_tidal"]="${EXAMPLES_DIR}/01_tidal_only/basic_tidal.yaml"
EXAMPLE_DESCRIPTIONS["basic_tidal"]="Pure tidal forcing with M2, S2, N2 constituents (elev_type=3, vel_type=3)"
EXAMPLE_CATEGORIES["basic_tidal"]="tidal"

EXAMPLES["extended_tidal"]="${EXAMPLES_DIR}/01_tidal_only/extended_tidal.yaml"
EXAMPLE_DESCRIPTIONS["extended_tidal"]="Tidal-only setup with refined timestep and additional namelist parameters"
EXAMPLE_CATEGORIES["extended_tidal"]="tidal"

EXAMPLES["tidal_with_potential"]="${EXAMPLES_DIR}/01_tidal_only/tidal_with_potential.yaml"
EXAMPLE_DESCRIPTIONS["tidal_with_potential"]="Tidal forcing with earth tidal potential and self-attraction loading"
EXAMPLE_CATEGORIES["tidal_with_potential"]="tidal"

# Hybrid examples
EXAMPLES["hybrid_elevation"]="${EXAMPLES_DIR}/02_hybrid/hybrid_elevation.yaml"
EXAMPLE_DESCRIPTIONS["hybrid_elevation"]="Combined tidal and external elevation data (elev_type=5)"
EXAMPLE_CATEGORIES["hybrid_elevation"]="hybrid"

EXAMPLES["full_hybrid"]="${EXAMPLES_DIR}/02_hybrid/full_hybrid.yaml"
EXAMPLE_DESCRIPTIONS["full_hybrid"]="Complete hybrid setup: tidal+external for elevation, velocity, temperature, salinity"
EXAMPLE_CATEGORIES["full_hybrid"]="hybrid"

# River examples
EXAMPLES["simple_river"]="${EXAMPLES_DIR}/03_river/simple_river.yaml"
EXAMPLE_DESCRIPTIONS["simple_river"]="River inflow (boundary 1) with constant flow/tracers, tidal ocean boundary"
EXAMPLE_CATEGORIES["simple_river"]="river"

EXAMPLES["multi_river"]="${EXAMPLES_DIR}/03_river/multi_river.yaml"
EXAMPLE_DESCRIPTIONS["multi_river"]="Multiple river boundaries with different flow rates and tracer properties"
EXAMPLE_CATEGORIES["multi_river"]="river"

# Nested examples
EXAMPLES["nested_with_tides"]="${EXAMPLES_DIR}/04_nested/nested_with_tides.yaml"
EXAMPLE_DESCRIPTIONS["nested_with_tides"]="Nested boundary conditions with relaxation and tidal forcing"
EXAMPLE_CATEGORIES["nested_with_tides"]="nested"

# Advanced examples
# Not working, need an example grid with more than one open boundary
# EXAMPLES["mixed_boundaries"]="${EXAMPLES_DIR}/05_advanced/mixed_boundaries.yaml"
# EXAMPLE_DESCRIPTIONS["mixed_boundaries"]="Mixed boundary types for complex domains"
# EXAMPLE_CATEGORIES["mixed_boundaries"]="advanced"

# Default settings
RUN_CATEGORY="all"
DRY_RUN=false
KEEP_OUTPUTS=false
SINGLE_EXAMPLE=""

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                RUN_CATEGORY="all"
                shift
                ;;
            --tidal)
                RUN_CATEGORY="tidal"
                shift
                ;;
            --hybrid)
                RUN_CATEGORY="hybrid"
                shift
                ;;
            --river)
                RUN_CATEGORY="river"
                shift
                ;;
            --nested)
                RUN_CATEGORY="nested"
                shift
                ;;
            --single)
                SINGLE_EXAMPLE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --keep-outputs)
                KEEP_OUTPUTS=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    echo "SCHISM Boundary Conditions Examples Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all              Run all examples (default)"
    echo "  --tidal            Run only tidal examples"
    echo "  --hybrid           Run only hybrid examples"
    echo "  --river            Run only river examples"
    echo "  --nested           Run only nested examples"
    echo "  --single <name>    Run single example by name"
    echo "  --dry-run          Show what would be run without executing"
    echo "  --keep-outputs     Keep output directories after run"
    echo "  --help             Show this help message"
    echo ""
    echo "Available examples:"
    for example in "${!EXAMPLES[@]}"; do
        printf "  %-25s %s\n" "$example" "${EXAMPLE_DESCRIPTIONS[$example]}"
    done
}

# Extract forcing tidal data from $PROJECT_ROOT/tests/schism/test_data/tpxo9-neaus.tar.gz
TIDAL_ARCHIVE="$PROJECT_ROOT/tests/schism/test_data/tpxo9-neaus.tar.gz"
TIDAL_DIR="$PROJECT_ROOT/tests/schism/test_data"

if [ ! -f "$TIDAL_ARCHIVE" ]; then
    echo "Error: Tidal data archive not found at $TIDAL_ARCHIVE" >&2
    exit 1
fi

if ! tar -xzf "$TIDAL_ARCHIVE" -C "$TIDAL_DIR"; then
    echo "Error: Failed to extract $TIDAL_ARCHIVE" >&2
    exit 1
fi

echo "Tidal data extracted successfully to $TIDAL_DIR"

# Get list of examples to run based on category
get_examples_to_run() {
    local examples_to_run=()

    if [[ -n "$SINGLE_EXAMPLE" ]]; then
        if [[ -n "${EXAMPLES[$SINGLE_EXAMPLE]}" ]]; then
            examples_to_run=("$SINGLE_EXAMPLE")
        else
            log_error "Example '$SINGLE_EXAMPLE' not found"
            exit 1
        fi
    else
        for example in "${!EXAMPLES[@]}"; do
            if [[ "$RUN_CATEGORY" == "all" ]] || [[ "${EXAMPLE_CATEGORIES[$example]}" == "$RUN_CATEGORY" ]]; then
                examples_to_run+=("$example")
            fi
        done
    fi

    printf '%s\n' "${examples_to_run[@]}"
}

# Run a single example
run_example() {
    local example_name="$1"
    local config_file="${EXAMPLES[$example_name]}"
    local description="${EXAMPLE_DESCRIPTIONS[$example_name]}"
    local output_dir="${BASE_OUTPUT_DIR}/${example_name}"

    log_header "Running Example: $example_name"
    log_info "Description: $description"
    log_info "Config file: $config_file"
    log_info "Output directory: $output_dir"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run $example_name"
        return 0
    fi

    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        return 1
    fi

    # Clean up previous run
    if [[ -d "$output_dir" ]]; then
        log_info "Cleaning up previous run directory: $output_dir"
        rm -rf "$output_dir"
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Step 1: Generate SCHISM configuration
    log_info "Generating SCHISM configuration..."
    # Get config file relative to project root
    local config_file_relative="${config_file#$PROJECT_ROOT/}"
    if ! (cd "$PROJECT_ROOT" && rompy schism "$config_file_relative"); then
        log_error "Failed to generate SCHISM configuration for $example_name"
        return 1
    fi

    # Find the generated directory (it should match the run_id in the config)
    local schism_dir=""
    case "$example_name" in
        "basic_tidal")
            schism_dir="schism_tidal_basic/basic_tidal_example"
            ;;
        "extended_tidal")
            schism_dir="schism_tidal_extended/extended_tidal_example"
            ;;
        "tidal_with_potential")
            schism_dir="schism_tidal_potential/tidal_potential_example"
            ;;
        "hybrid_elevation")
            schism_dir="schism_hybrid_elevation/hybrid_elevation_example"
            ;;
        "full_hybrid")
            schism_dir="schism_full_hybrid/full_hybrid_example"
            ;;
        "simple_river")
            schism_dir="schism_simple_river/simple_river_example"
            ;;
        "multi_river")
            schism_dir="schism_multi_river/multi_river_example"
            ;;
        "nested_with_tides")
            schism_dir="schism_nested_with_tides/nested_with_tides_example"
            ;;
        "mixed_boundaries")
            schism_dir="schism_mixed_boundaries/mixed_boundaries_example"
            ;;
    esac
    schism_dir="$PROJECT_ROOT/$schism_dir"

    if [[ ! -d "$schism_dir" ]]; then
        log_error "Generated SCHISM directory not found: $schism_dir"
        return 1
    fi

    # Copy the station.in file if it exists to the schism_dir
    local station_file="$PROJECT_ROOT/notebooks/schism/station.in"
    if [[ -f "$station_file" ]]; then
        log_info "Copying station.in file to SCHISM directory"
        cp "$station_file" "$schism_dir/"
    else
        log_warning "station.in file not found, skipping copy"
    fi

    # Step 2: Inspect directory structure
    log_info "Inspecting generated directory structure..."
    docker run -v "$schism_dir:/tmp/schism:Z" schism bash -c "ls /tmp/schism/ > /dev/null && echo 'Files in directory:' && find /tmp/schism -type f -name '*.in' -o -name '*.gr3' -o -name '*.nc'"

    # Step 3: Run SCHISM simulation
    log_info "Running SCHISM simulation..."
    if docker run -v "$schism_dir:/tmp/schism:Z" schism bash -c "cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_${SCHISM_VERSION} 4"; then
        log_success "SCHISM simulation completed successfully for $example_name"

        # Check for output files
        if docker run -v "$schism_dir:/tmp/schism:Z" schism bash -c "ls -la /tmp/schism/outputs/*.nc" &>/dev/null; then
            log_success "Output files generated successfully"
        else
            log_warning "No output files found - simulation may have failed"
            return 1
        fi
    else
        log_error "SCHISM simulation failed for $example_name"
        return 1
    fi

    # Step 4: Move results to organized output directory
    if [[ -d "$schism_dir" ]]; then
        mv "$schism_dir" "$output_dir/"
        log_info "Results moved to: $output_dir/"
    fi

    return 0
}

# Main execution function
main() {
    parse_args "$@"

    log_header "SCHISM Boundary Conditions Examples Test Runner"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Examples directory: $EXAMPLES_DIR"
    log_info "SCHISM Version: $SCHISM_VERSION"
    log_info "Run category: $RUN_CATEGORY"
    log_info "Dry run: $DRY_RUN"
    log_info "Keep outputs: $KEEP_OUTPUTS"

    # Get examples to run
    mapfile -t examples_to_run < <(get_examples_to_run)

    if [[ ${#examples_to_run[@]} -eq 0 ]]; then
        log_warning "No examples found matching criteria"
        exit 0
    fi

    log_info "Examples to run: ${#examples_to_run[@]}"
    for example in "${examples_to_run[@]}"; do
        log_info "  - $example: ${EXAMPLE_DESCRIPTIONS[$example]}"
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run complete - no examples were actually executed"
        exit 0
    fi

    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"

    # Track results
    local successful_runs=()
    local failed_runs=()
    local start_time=$(date +%s)

    # Run examples
    for example in "${examples_to_run[@]}"; do
        if run_example "$example"; then
            successful_runs+=("$example")
        else
            failed_runs+=("$example")
        fi
        echo ""  # Add spacing between examples
    done

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_header "Test Run Summary"
    log_info "Total duration: ${duration} seconds"
    log_info "Total examples: ${#examples_to_run[@]}"
    log_success "Successful runs: ${#successful_runs[@]}"

    if [[ ${#successful_runs[@]} -gt 0 ]]; then
        for example in "${successful_runs[@]}"; do
            log_success "  ✓ $example"
        done
    fi

    if [[ ${#failed_runs[@]} -gt 0 ]]; then
        log_error "Failed runs: ${#failed_runs[@]}"
        for example in "${failed_runs[@]}"; do
            log_error "  ✗ $example"
        done
    fi

    # Clean up if requested
    if [[ "$KEEP_OUTPUTS" == "false" ]] && [[ ${#failed_runs[@]} -eq 0 ]]; then
        log_info "Cleaning up output directories..."
        rm -rf "$BASE_OUTPUT_DIR"
        log_info "Cleanup complete"
    else
        log_info "Output directories preserved in: $BASE_OUTPUT_DIR"
    fi

    # Exit with error if any runs failed
    if [[ ${#failed_runs[@]} -gt 0 ]]; then
        exit 1
    fi

    log_success "All boundary condition examples completed successfully!"
}

# Check prerequisites
check_prerequisites() {
    # Check if rompy command is available
    if ! command -v rompy &> /dev/null; then
        log_error "rompy command not found. Please install ROMPY first."
        exit 1
    fi

    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        log_error "docker command not found. Please install Docker first."
        exit 1
    fi

    # Check if SCHISM docker image is available
    if ! docker image inspect schism &> /dev/null; then
        log_error "SCHISM Docker image not found. Please build or pull the SCHISM image."
        exit 1
    fi

    # Check if examples directory exists
    if [[ ! -d "$EXAMPLES_DIR" ]]; then
        log_error "Examples directory not found: $EXAMPLES_DIR"
        log_info "Project root detected: $PROJECT_ROOT"
        log_info "Please ensure the boundary condition examples are available"
        exit 1
    fi
}

# Run prerequisites check and main function
check_prerequisites
main "$@"
