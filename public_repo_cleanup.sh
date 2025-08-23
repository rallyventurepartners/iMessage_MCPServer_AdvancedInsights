#!/bin/bash

# Enhanced cleanup script for public repository release
# This script removes backup directories, cleans up old documentation,
# and ensures the project structure is clean for public release

echo "=== Starting Enhanced Cleanup for Public Repository ==="

# Create a minimal backup of just the core files (not previous backups)
FINAL_BACKUP_DIR="final_backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating minimal backup of core files to $FINAL_BACKUP_DIR"
mkdir -p "$FINAL_BACKUP_DIR"

# Only back up essential files to minimize storage
cp -R src scripts tests mcp_server_*.py README.md LICENSE requirements*.txt .gitignore .pre-commit-config.yaml "$FINAL_BACKUP_DIR" 2>/dev/null || true

# Remove all project backup directories
echo "Removing all backup directories..."
rm -rf project_backup_*

# Remove sample implementation directories
echo "Removing sample implementation directories..."
rm -rf sample_implementation*

# Cleaning up documentation
echo "Cleaning up documentation..."
mkdir -p docs/archive

# Keep only essential documentation, move others to archive
essential_docs=(
  "USER_GUIDE.md"
  "DEVELOPER_GUIDE.md"
  "CONTRIBUTING.md"
  "CLAUDE_DESKTOP_INTEGRATION.md"
  "CODE_REVIEW_IMPLEMENTATION.md"
  "PRODUCTION_READINESS.md"
)

# Move non-essential docs to archive
for doc in docs/*.md; do
  filename=$(basename "$doc")
  is_essential=false
  
  for essential in "${essential_docs[@]}"; do
    if [ "$filename" == "$essential" ]; then
      is_essential=true
      break
    fi
  done
  
  if [ "$is_essential" == "false" ]; then
    mv "$doc" docs/archive/
    echo "Archived: $doc"
  fi
done

# Compress the archived docs
echo "Compressing archived documentation..."
cd docs
tar -czf old_docs.tar.gz archive
rm -rf archive
cd ..

# Remove any temporary or intermediary files
echo "Removing temporary and intermediary files..."
rm -f basic_performance_test_results.json
rm -f code_review_analysis.md
rm -f code_review_improvements.md
rm -f code_review_results.md
rm -f implementation_plan.md
rm -f file_inventory.md
rm -f file_structure_analysis.md

# Clean up scripts directory
echo "Cleaning up scripts directory..."
mkdir -p scripts_archive
mv scripts_archive/* scripts/ 2>/dev/null || true
rmdir scripts_archive

# Remove old scripts that aren't needed for public release
old_scripts=(
  "fix_permissions.py"
  "setup_test_sharding.py"
  "verify_production_structure.sh"
  "verify_cleanup.sh"
  "production_cleanup.sh"
)

for script in "${old_scripts[@]}"; do
  if [ -f "scripts/$script" ]; then
    rm "scripts/$script"
    echo "Removed: scripts/$script"
  fi
  if [ -f "$script" ]; then
    rm "$script"
    echo "Removed: $script"
  fi
done

# Remove any Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".DS_Store" -delete

# Update README.md for public release if needed
echo "Final preparation complete. Project is now cleaned up for public repository release."
echo "A final backup was created in $FINAL_BACKUP_DIR"
echo ""
echo "Next steps:"
echo "1. Review the project structure to ensure it's as expected"
echo "2. Update README.md for public audience if needed"
echo "3. Push to public repository"
