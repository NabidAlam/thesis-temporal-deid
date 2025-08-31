#!/bin/bash

# =============================================================================
# IMPROVED TEMPORAL DEIDENTIFICATION PIPELINE RUNNER
# =============================================================================
# This script provides a comprehensive interface for running various deidentification
# pipelines with different configurations and datasets.
# =============================================================================

# Exit immediately if a command fails
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --pipeline TYPE          Pipeline type to run:"
    echo "                            - integrated_temporal (default)"
    echo "                            - focused_sam"
    echo "                            - standard_sam"
    echo "                            - samurai_only"
    echo "                            - tsp_sam_only"
    echo ""
    echo "  --dataset NAME            Dataset to process:"
    echo "                            - ted (default)"
    echo "                            - davis2017"
    echo "                            - team_ten"
    echo "                            - tragic_talkers"
    echo ""
    echo "  --max-frames N            Maximum frames to process per video (default: 50)"
    echo "  --force                   Force overwrite existing output directories"
    echo "  --debug                   Enable debug mode with mask visualizations"
    echo "  --wandb                   Enable Weights & Biases tracking"
    echo "  --experiment-name NAME    Name for W&B experiment"
    echo "  --list-videos             List available videos in dataset"
    echo "  --help                    Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --pipeline integrated_temporal --dataset ted --max-frames 100"
    echo "  $0 --pipeline focused_sam --dataset davis2017 --debug"
    echo "  $0 --list-videos --dataset ted"
    echo ""
}

# Function to list available videos
list_videos() {
    local dataset=$1
    local video_dir="input/$dataset"
    
    if [ ! -d "$video_dir" ]; then
        print_error "Dataset directory not found: $video_dir"
        exit 1
    fi
    
    print_header "Available videos in $dataset dataset:"
    
    local count=0
    for video in "$video_dir"/*.mp4; do
        if [ -f "$video" ]; then
            local size=$(du -h "$video" | cut -f1)
            local duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null | cut -d. -f1)
            local duration_str=""
            
            if [ ! -z "$duration" ] && [ "$duration" != "N/A" ]; then
                duration_str=" (${duration}s)"
            fi
            
            echo -e "  ${CYAN}$(basename "$video")${NC} - ${size}${duration_str}"
            ((count++))
        fi
    done
    
    if [ $count -eq 0 ]; then
        print_warning "No .mp4 files found in $video_dir"
    else
        echo ""
        print_status "Found $count video(s) in $dataset dataset"
    fi
}

# Function to run integrated temporal pipeline
run_integrated_temporal() {
    local video_path=$1
    local output_dir=$2
    local max_frames=$3
    local debug_flag=$4
    local wandb_flag=$5
    local experiment_name=$6
    
    local video_name=$(basename "$video_path" .mp4)
    local full_output_dir="$output_dir/$video_name"
    
    print_status "Running integrated temporal pipeline on $video_name..."
    
    # Build command
    local cmd="python integrated_temporal_pipeline.py"
    cmd="$cmd \"$video_path\""
    cmd="$cmd \"$full_output_dir\""
    cmd="$cmd --max-frames $max_frames"
    
    if [ "$debug_flag" = "true" ]; then
        cmd="$cmd --debug"
    fi
    
    if [ "$wandb_flag" = "true" ]; then
        cmd="$cmd --wandb"
        if [ ! -z "$experiment_name" ]; then
            cmd="$cmd --experiment-name \"$experiment_name\""
        fi
    fi
    
    if [ "$force_flag" = "true" ]; then
        cmd="$cmd --force"
    fi
    
    print_status "Command: $cmd"
    echo ""
    
    # Execute command
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_status "Successfully processed $video_name"
    else
        print_error "Failed to process $video_name"
        return 1
    fi
}

# Function to run focused SAM baseline
run_focused_sam() {
    local video_path=$1
    local output_dir=$2
    local max_frames=$3
    
    local video_name=$(basename "$video_path" .mp4)
    local full_output_dir="$output_dir/$video_name"
    
    print_status "Running focused SAM baseline on $video_name..."
    
    # Create output directory
    mkdir -p "$full_output_dir"
    
    # Run focused SAM
    python run_focused_sam_baseline.py \
        --input "$video_path" \
        --output "$full_output_dir" \
        --max-frames $max_frames
    
    if [ $? -eq 0 ]; then
        print_status "Successfully processed $video_name with focused SAM"
    else
        print_error "Failed to process $video_name with focused SAM"
        return 1
    fi
}

# Function to run standard SAM
run_standard_sam() {
    local video_path=$1
    local output_dir=$2
    local max_frames=$3
    
    local video_name=$(basename "$video_path" .mp4)
    local full_output_dir="$output_dir/$video_name"
    
    print_status "Running standard SAM on $video_name..."
    
    # Create output directory
    mkdir -p "$full_output_dir"
    
    # Run standard SAM
    python run_working_standard_sam.py \
        --input "$video_path" \
        --output "$full_output_dir" \
        --max-frames $max_frames
    
    if [ $? -eq 0 ]; then
        print_status "Successfully processed $video_name with standard SAM"
    else
        print_error "Failed to process $video_name with standard SAM"
        return 1
    fi
}

# Function to run SAMURAI only
run_samurai_only() {
    local video_path=$1
    local output_dir=$2
    local max_frames=$3
    
    local video_name=$(basename "$video_path" .mp4)
    local full_output_dir="$output_dir/$video_name"
    
    print_status "Running SAMURAI only on $video_name..."
    
    # Create output directory
    mkdir -p "$full_output_dir"
    
    # Run SAMURAI
    python samurai_runner.py \
        --input "$video_path" \
        --output "$full_output_dir" \
        --max-frames $max_frames
    
    if [ $? -eq 0 ]; then
        print_status "Successfully processed $video_name with SAMURAI"
    else
        print_error "Failed to process $video_name with SAMURAI"
        return 1
    fi
}

# Function to run TSP-SAM only
run_tsp_sam_only() {
    local video_path=$1
    local output_dir=$2
    local max_frames=$3
    
    local video_name=$(basename "$video_path" .mp4)
    local full_output_dir="$output_dir/$video_name"
    
    print_status "Running TSP-SAM only on $video_name..."
    
    # Create output directory
    mkdir -p "$full_output_dir"
    
    # Run TSP-SAM
    python temporal/tsp_sam_complete.py \
        --input "$video_path" \
        --output "$full_output_dir" \
        --max-frames $max_frames
    
    if [ $? -eq 0 ]; then
        print_status "Successfully processed $video_name with TSP-SAM"
    else
        print_error "Failed to process $video_name with TSP-SAM"
        return 1
    fi
}

# Function to process all videos in dataset
process_dataset() {
    local pipeline_type=$1
    local dataset=$2
    local max_frames=$3
    local debug_flag=$4
    local wandb_flag=$5
    local experiment_name=$6
    
    local video_dir="input/$dataset"
    local output_dir="output/$pipeline_type/$dataset"
    
    if [ ! -d "$video_dir" ]; then
        print_error "Dataset directory not found: $video_dir"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    print_header "Processing $dataset dataset with $pipeline_type pipeline"
    print_status "Input directory: $video_dir"
    print_status "Output directory: $output_dir"
    print_status "Max frames per video: $max_frames"
    echo ""
    
    local count=0
    local success_count=0
    
    for video in "$video_dir"/*.mp4; do
        if [ -f "$video" ]; then
            ((count++))
            print_status "Processing video $count: $(basename "$video")"
            
            case $pipeline_type in
                "integrated_temporal")
                    if run_integrated_temporal "$video" "$output_dir" $max_frames "$debug_flag" "$wandb_flag" "$experiment_name"; then
                        ((success_count++))
                    fi
                    ;;
                "focused_sam")
                    if run_focused_sam "$video" "$output_dir" $max_frames; then
                        ((success_count++))
                    fi
                    ;;
                "standard_sam")
                    if run_standard_sam "$video" "$output_dir" $max_frames; then
                        ((success_count++))
                    fi
                    ;;
                "samurai_only")
                    if run_samurai_only "$video" "$output_dir" $max_frames; then
                        ((success_count++))
                    fi
                    ;;
                "tsp_sam_only")
                    if run_tsp_sam_only "$video" "$output_dir" $max_frames; then
                        ((success_count++))
                    fi
                    ;;
                *)
                    print_error "Unknown pipeline type: $pipeline_type"
                    exit 1
                    ;;
            esac
            
            echo ""
        fi
    done
    
    if [ $count -eq 0 ]; then
        print_warning "No .mp4 files found in $video_dir"
        return 0
    fi
    
    print_header "Processing Complete"
    print_status "Total videos: $count"
    print_status "Successful: $success_count"
    print_status "Failed: $((count - success_count))"
    
    if [ $success_count -eq $count ]; then
        print_status "All videos processed successfully! 🎉"
    else
        print_warning "Some videos failed to process. Check logs for details."
    fi
}

# Main script
main() {
    # Default values
    PIPELINE_TYPE="integrated_temporal"
    DATASET="ted"
    MAX_FRAMES=50
    FORCE_FLAG="false"
    DEBUG_FLAG="false"
    WANDB_FLAG="false"
    EXPERIMENT_NAME=""
    LIST_VIDEOS="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pipeline)
                PIPELINE_TYPE="$2"
                shift 2
                ;;
            --dataset)
                DATASET="$2"
                shift 2
                ;;
            --max-frames)
                MAX_FRAMES="$2"
                shift 2
                ;;
            --force)
                FORCE_FLAG="true"
                shift
                ;;
            --debug)
                DEBUG_FLAG="true"
                shift
                ;;
            --wandb)
                WANDB_FLAG="true"
                shift
                ;;
            --experiment-name)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --list-videos)
                LIST_VIDEOS="true"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate pipeline type
    case $PIPELINE_TYPE in
        "integrated_temporal"|"focused_sam"|"standard_sam"|"samurai_only"|"tsp_sam_only")
            ;;
        *)
            print_error "Invalid pipeline type: $PIPELINE_TYPE"
            show_usage
            exit 1
            ;;
    esac
    
    # Validate dataset
    case $DATASET in
        "ted"|"davis2017"|"team_ten"|"tragic_talkers")
            ;;
        *)
            print_error "Invalid dataset: $DATASET"
            show_usage
            exit 1
            ;;
    esac
    
    # Validate max frames
    if ! [[ "$MAX_FRAMES" =~ ^[0-9]+$ ]] || [ "$MAX_FRAMES" -lt 1 ]; then
        print_error "Max frames must be a positive integer"
        exit 1
    fi
    
    # Set global force flag
    force_flag=$FORCE_FLAG
    
    # Show configuration
    print_header "Configuration"
    print_status "Pipeline: $PIPELINE_TYPE"
    print_status "Dataset: $DATASET"
    print_status "Max frames: $MAX_FRAMES"
    print_status "Force overwrite: $FORCE_FLAG"
    print_status "Debug mode: $DEBUG_FLAG"
    print_status "W&B tracking: $WANDB_FLAG"
    if [ ! -z "$EXPERIMENT_NAME" ]; then
        print_status "Experiment name: $EXPERIMENT_NAME"
    fi
    echo ""
    
    # Check if just listing videos
    if [ "$LIST_VIDEOS" = "true" ]; then
        list_videos "$DATASET"
        exit 0
    fi
    
    # Check if conda environment is activated
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_warning "Conda environment not detected. Make sure to activate 'marbeit' environment for CUDA support."
        echo ""
    else
        print_status "Using conda environment: $CONDA_DEFAULT_ENV"
        echo ""
    fi
    
    # Process dataset
    process_dataset "$PIPELINE_TYPE" "$DATASET" "$MAX_FRAMES" "$DEBUG_FLAG" "$WANDB_FLAG" "$EXPERIMENT_NAME"
}

# Run main function with all arguments
main "$@"
